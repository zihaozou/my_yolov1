import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

##Pre-Trained VGG11
class pretrainedVGG11(nn.Module):
    def __init__(self):
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
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7,1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(p= 0.5),
            nn.Linear(1024,1000),
        )
        self.detector=nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Dropout(p= 0.6),
            nn.Linear(128*7*7,1024),
            nn.Tanh(),
            nn.Linear(1024,1470)
        )
        self.weightInit()
    def forward(self,x,cls=False):
        x = self.features(x)
        if cls:
            x=self.classifier(x)
            x=torch.softmax(x,3)
        else:
            x=self.detector(x).view(-1,7,7,30)
            x=torch.cat((torch.sigmoid(x[:,:,:,:10]),torch.softmax(x[:,:,:,10:],3)),3)
        return x
    def weightInit(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


