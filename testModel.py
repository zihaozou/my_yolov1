from torch.autograd.variable import Variable
from yoloLoss import yoloLoss
from myVOCDataSet import loadVOCTrainDataSet, loadVOCValDataSet
from myYOLOModel import loadvgg11
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as op
import cv2
import torchvision.transforms.functional as tfFunc
from myVOCDataSet import VOCLIST
model=loadvgg11(pretrain=False)
model.load_state_dict(torch.load('vggYolo.pth'))
model.eval()
img=cv2.imread('000019.jpg')

cv2.imshow('image',img)

img=tfFunc.to_tensor(img)
tempimg=tfFunc.resize(img,[448,448])
pred=model(tempimg.unsqueeze(0)).squeeze()

temp1=(torch.arange(7)*64).unsqueeze(1).expand(-1,7).unsqueeze(-1).expand(-1,-1,2)
temp2=(torch.arange(7)*64).unsqueeze(0).expand(7,-1).unsqueeze(-1).expand(-1,-1,2)
pred[:,:,[0,1,5,6]]=pred[:,:,[0,1,5,6]]*64+(torch.cat((temp1,temp2),2))
pred[:,:,[2,3,7,8]]=pred[:,:,[2,3,7,8]]*448
gridList=pred.view(-1,30)
boxList=gridList[:,:10].view(-1,2,5)
classList=gridList[:,10:]
print(boxList)
_,i=torch.max(boxList[:,:,4],dim=1)
boxList=boxList[torch.arange(boxList.size(0)),i]
xyMinList=(boxList[:,[0,1]]-(boxList[:,[2,3]]/2))
xyMaxList=(boxList[:,[0,1]]+(boxList[:,[2,3]]/2))
cornerList=torch.cat((xyMinList,xyMaxList),1)
cornerList[:,[0,2]]=(cornerList[:,[0,2]]*img.size(1)/448)
cornerList[:,[1,3]]=(cornerList[:,[1,3]]*img.size(2)/448)
cornerList=torch.cat((cornerList.type(torch.IntTensor),boxList[:,4].unsqueeze(-1)),1)
_,classList=classList.max(1)
labelList=[]
for v in classList:
    labelList.append(VOCLIST[v])
print(labelList)



