import torch
import numpy as np
import cv2
from myVOCDataSet import VOCLIST
from yoloLoss import compute_iou
import numpy as np



def drawBBoxes(img,pred,thrhd,gridSize,inputSize):
    imgArr=np.array(img.permute(1,2,0))
    temp1=(torch.arange(7)*gridSize).unsqueeze(1).expand(-1,7).unsqueeze(-1)
    temp2=(torch.arange(7)*gridSize).unsqueeze(0).expand(7,-1).unsqueeze(-1)
    pred[:,:,[0,1,5,6]]=pred[:,:,[0,1,5,6]]*gridSize+(torch.cat((temp1,temp2,temp1,temp2),2))
    pred[:,:,[2,3,7,8]]=pred[:,:,[2,3,7,8]]*inputSize
    gridList=pred.view(-1,30)
    boxList=gridList[:,:10].view(-1,2,5)
    classList=gridList[:,10:]-10
    _,classList=classList.max(1)
    _,i=torch.max(boxList[:,:,4],dim=1)
    boxList=boxList[torch.arange(boxList.size(0)),i]
    boxList=torch.cat((boxList,classList.unsqueeze(-1)),1)

    mask=(boxList[:,4]>thrhd).unsqueeze(-1).expand_as(boxList)
    boxList=boxList[mask].view(-1,6)
    xyMinList=(boxList[:,[0,1]]-(boxList[:,[2,3]]/2))
    xyMaxList=(boxList[:,[0,1]]+(boxList[:,[2,3]]/2))
    cornerList=torch.cat((xyMinList,xyMaxList),1)
    cornerList[:,[0,2]]=(cornerList[:,[0,2]]*img.size(1)/inputSize)
    cornerList[:,[1,3]]=(cornerList[:,[1,3]]*img.size(2)/inputSize)
    cornerList=torch.cat((cornerList.type(torch.IntTensor),boxList[:,4:]),1)
    SolidCorner=torch.Tensor()
    while len(cornerList)!=0:
        i=torch.max(cornerList[:,4],dim=0)[1]
        highest=cornerList[i,:]
        cornerList=cornerList[torch.arange(cornerList.size(0))!=i,:]
        deleteList=torch.Tensor()
        for x in range(cornerList.size(0)):
            comparing=cornerList[x,:]
            if compute_iou(highest[:4].unsqueeze(0),comparing[:4].unsqueeze(0))>thrhd:
                deleteList=torch.cat((deleteList,comparing.unsqueeze(0)),0)
        cornerList=cornerList[cornerList!=deleteList]
        SolidCorner=torch.cat((SolidCorner,highest.unsqueeze(0)),0)
    for i,r in enumerate(SolidCorner):
        imgArr=cv2.rectangle(imgArr,(int(r[1]),int(r[0])),(int(r[3]),int(r[2])),(255,0,0),1)
        imgArr=cv2.putText(imgArr,
        f'{r[4].item():1.2f}'+':'+VOCLIST[int(r[5])],
        (int(r[1]),int(r[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (209, 80, 0, 255),
        3)
    return img,SolidCorner







