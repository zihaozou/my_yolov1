import random 
from torch import nn
from myYOLOModel import pretrainedVGG11
from torch.autograd.variable import Variable
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR100
import argparse
import os
import numpy as np
import torch.optim as op
from PIL import Image, ImageEnhance, ImageFilter
parser = argparse.ArgumentParser(description='PyTorch Yolo Classification Training')
parser.add_argument('--data-dir','-dd',nargs='?',const=1,default='../data',type=str)


class imgAug(object):
    changeFactor=[0.6,1,1.2,1.3,1.5,1.8]
    def __init__(self) -> None:
        super().__init__()
    def __call__(self,img):
        if random.random()<.5:
            img=self.randomBrightness(img)
            img=self.randomContrast(img)
            img=self.randomFlip(img)
            img=self.randomSatuation(img)
            img=self.randomSharporBlur(img)
        return img
    def randomFlip(self,img:Image.Image):
        if random.random()<.5:
            img= img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    def randomBrightness(self,img:Image.Image):
        b=random.choice(self.changeFactor)
        return ImageEnhance.Brightness(img).enhance(b)
    def randomContrast(self,img:Image.Image):
        c=random.choice(self.changeFactor)
        return ImageEnhance.Contrast(img).enhance(c)
    def randomSharporBlur(self,img:Image.Image):
        if random.random()<.5:
            img=img.filter(ImageFilter.GaussianBlur(1.5))
        else:
            s=random.choice(self.changeFactor)
            img=ImageEnhance.Sharpness(img).enhance(s)
        return img 
    def randomSatuation(self,img:Image.Image):
        s=random.choice(self.changeFactor)
        return ImageEnhance.Color(img).enhance(s)



def main():
    args=parser.parse_args()
    dataDir=args.data_dir
    batchSize=args.batch_size
    numWorkers=args.num_workers
    epochs=args.epochs
    weightDir=args.weight_dir
    if args.gpu:
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device='cpu'
    print('using device:'+device)


    download=False if os.path.isdir(dataDir+'cifar-100-python') else True
    cifarTrainDataset=CIFAR100(root=dataDir,
                                train=True,
                                download=download,
                                transform=imgAug()
                                )
    model=pretrainedVGG11().to(device)
    if args.weighted!=None:
        model.load_state_dict(torch.load(weightDir))
    optimizer=op.AdamW(model.parameters())
    criterion=CrossEntropyLoss()
    trainLoader=DataLoader(cifarTrainDataset,batch_size=batchSize,shuffle=True,num_workers=numWorkers)
    
    for e in range(epochs):
        model.train()
        print('epoch '+str(e))
        for (image,target) in enumerate(trainLoader):
            image=Variable(image).to(device)
            target=Variable()





if __name__ == '__main__':
    main()
