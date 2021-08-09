from torch.autograd.variable import Variable
from yoloLoss import yoloLoss
from myVOCDataSet import loadVOCTrainDataSet, loadVOCValDataSet
from myYOLOModel import loadvgg11
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as op
model=loadvgg11(pretrain=False)
model.load_state_dict(torch.load('my_yolov1/vggYolo.pth'))
criterion=yoloLoss(7,2,5,.5)
set=loadVOCValDataSet()
model.eval()
for i,(img,tg) in enumerate(set):
    if i>20:
        break
    img=Variable(torch.unsqueeze(img,0))
    tg=Variable(torch.unsqueeze(tg,0))
    pred=model(img)
    loss,a,b,c,d,e=criterion(pred,tg,val=True)
    print(a.data,b.data,c.data,d.data,e.data)


