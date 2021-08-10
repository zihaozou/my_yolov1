#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#device='cpu'
class yoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj,cloud=False,cf=1,csf=1,nc=1):
        super(yoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.cf=cf
        self.csf=csf
        self.nc=nc
        if cloud:
            self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device='cpu'
    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)#
        M = box2.size(0)

        lt = torch.max(#计算右上角盒子的顶点
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(#计算左下角盒子的顶点
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]左下减右上
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]算出相交面积

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def forward(self,pred_tensor,target_tensor,val=False):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.BoolTensor(noo_pred.size()).to(self.device)
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,reduction='sum')

        #compute contain obj loss
        coo_response_mask = torch.BoolTensor(box_target.size()).to(self.device)
        coo_response_mask.zero_()
        coo_not_response_mask = torch.BoolTensor(box_target.size()).to(self.device)
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).to(self.device)
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size())).to(self.device)
            box1_xyxy[:,:2] = box1[:,:2]*64. -0.5*(box1[:,2:4]*448)
            box1_xyxy[:,2:4] = box1[:,:2]*64 +0.5*(box1[:,2:4]*448)
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size())).to(self.device)
            box2_xyxy[:,:2] = box2[:,:2]*64 -0.5*(box2[:,2:4]*448)
            box2_xyxy[:,2:4] = box2[:,:2]*64 +0.5*(box2[:,2:4]*448)
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4]).to(self.device)] = (max_iou).data
        box_target_iou = Variable(box_target_iou).to(self.device)
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],reduction='sum')
        loc_loss+= F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),reduction='sum')
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],reduction='sum')

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,reduction='sum')
        total_loss=(self.l_coord*loc_loss + contain_loss*self.cf + self.nc*not_contain_loss+self.l_noobj*nooobj_loss + class_loss*self.csf)/N
        if val==True:
            return total_loss,loc_loss/N,contain_loss/N,not_contain_loss/N,nooobj_loss/N,class_loss/N
        return total_loss




