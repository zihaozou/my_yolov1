import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.utils.model_zoo as mz
cloudInputDir='/input0'
class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

##Pre-Trained VGG11
class pretrainedVGG11(nn.Module):
    pretrainURL='https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
    def __init__(self,do):
        super(pretrainedVGG11,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7,4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(p= do),
            nn.Linear(4096,7*7*30),
        )
        
    def forward(self,x):
        x = self.classifier(self.features(x))
        x=x.view(-1,7,7,30)
        new=x.detach().clone()
        new[:,:,:,:10]=torch.sigmoid(x[:,:,:,:10])
        new[:,:,:,10:]=torch.softmax(x[:,:,:,10:],dim=3)
        return new





def loadvgg11(cloud=False,pretrain=False,do=0.6):
    vgg=pretrainedVGG11(do)
    if pretrain:
        state=vgg.state_dict()
        if cloud:
            pretrained=torch.load(cloudInputDir+'/vgg11_bn-6002323d.pth')
        else:
            pretrained=mz.load_url(vgg.pretrainURL)
        for k in pretrained.keys():
            if k in state.keys() and k.startswith('features'):
                state[k] = pretrained[k]
        vgg.load_state_dict(state)
    return vgg

