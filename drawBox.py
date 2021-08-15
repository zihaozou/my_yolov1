from PIL import ImageFont
import torch
from PIL import Image,ImageDraw
from yoloLoss import compute_iou
from torchvision.transforms.functional import to_tensor
VOCLIST=['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

def image_dpi_resize(image:Image.Image):
    """
    Rescaling image to 300dpi while resizing
    :param image: An image
    :return: A rescaled image
    """
    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = image.resize(size, Image.ANTIALIAS)
    return image_resize



def drawBBoxes(imgArr:Image.Image,pred:torch.Tensor,thrhd,gridSize,inputSize,device):
    imgArr=image_dpi_resize(imgArr)
    temp1=(torch.arange(7)*gridSize).unsqueeze(1).expand(-1,7).unsqueeze(-1).to(device)
    temp2=(torch.arange(7)*gridSize).unsqueeze(0).expand(7,-1).unsqueeze(-1).to(device)
    pred=pred.squeeze()
    pred[:,:,[0,1,5,6]]=pred[:,:,[0,1,5,6]]*gridSize+(torch.cat((temp1,temp2,temp1,temp2),2))
    pred[:,:,[2,3,7,8]]=pred[:,:,[2,3,7,8]]*inputSize
    gridList=pred.view(-1,30)
    boxList=gridList[:,:10].view(-1,2,5)
    classList=gridList[:,10:]-10
    _,classList=classList.max(1)
    _,i=torch.max(boxList[:,:,4],dim=1)
    boxList=boxList[torch.arange(boxList.size(0),device=device),i]
    boxList=torch.cat((boxList,classList.unsqueeze(-1)),1)

    mask=(boxList[:,4]>thrhd).unsqueeze(-1).expand_as(boxList)
    boxList=boxList[mask].view(-1,6)
    xyMinList=(boxList[:,[0,1]]-(boxList[:,[2,3]]/2))
    xyMaxList=(boxList[:,[0,1]]+(boxList[:,[2,3]]/2))
    cornerList=torch.cat((xyMinList,xyMaxList),1)
    cornerList[:,[0,2]]=(cornerList[:,[0,2]]*imgArr.size[1]/inputSize)
    cornerList[:,[1,3]]=(cornerList[:,[1,3]]*imgArr.size[0]/inputSize)
    cornerList=torch.cat((cornerList.type(torch.IntTensor).to(device),boxList[:,4:]),1)
    mask=torch.logical_and(cornerList[:,0]>=0,cornerList[:,1]>=0)
    mask=torch.logical_and(torch.logical_and(cornerList[:,2]<=imgArr.size[1],cornerList[:,3]<=imgArr.size[0]),mask)
    mask=mask.unsqueeze(-1).expand_as(cornerList)
    cornerList=cornerList[mask].view(-1,6)
    SolidCorner=torch.Tensor().to(device)
    while len(cornerList)!=0:
        _,i=torch.max(cornerList[:,4],dim=0)
        highest=cornerList[i,:]
        cornerList=cornerList[torch.arange(cornerList.size(0),device=device)!=i,:]
        if len(cornerList)!=0:
            deleteList=torch.BoolTensor(cornerList.size(0)).to(device).zero_()
            for x in range(cornerList.size(0)):
                comparing=cornerList[x,:]
                if compute_iou(highest[:4].unsqueeze(0),comparing[:4].unsqueeze(0))<thrhd:
                    deleteList[x]=True
            deleteList=deleteList.unsqueeze(-1).expand_as(cornerList)
            cornerList=cornerList[deleteList].view(-1,6)
        SolidCorner=torch.cat((SolidCorner,highest.unsqueeze(0)),0)
    drawer=ImageDraw.Draw(imgArr)
    font = ImageFont.truetype("UbuntuMono-R.ttf",15)
    for i,r in enumerate(SolidCorner):
        drawer.rectangle((r[1],r[0],r[3],r[2]),outline='red')
        drawer.text((r[1],r[0]),VOCLIST[int(r[5])],font=font)
    img=to_tensor(imgArr)
    return img,SolidCorner







