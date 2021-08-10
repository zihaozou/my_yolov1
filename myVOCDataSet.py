import random
import numpy
import torch
from torchvision import datasets
import torchvision.transforms.functional as tfFunc
import torchvision.transforms as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from torch.utils.data.dataloader import DataLoader
CloudInputDir='/input0'
VOC_CLASSES = {    # always index 0
    'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3,
    'bottle':4, 'bus':5, 'car':6, 'cat':7, 'chair':8,
    'cow':9, 'diningtable':10, 'dog':11, 'horse':12,
    'motorbike':13, 'person':14, 'pottedplant':15,
    'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}
VOCLIST=['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
class myVOCtransform(object):
    def __init__(self,classlist,S=7,B=2,C=20,inputSize=448,train=False):
        super().__init__()
        self.S=S
        self.B=B
        self.C=C
        self.inputSize=inputSize
        self.classlist=classlist
        self.train=train
        self.gray=tf.RandomGrayscale(p=0.5)
    def __call__(self, img,anno):
        random.seed()
        
        target=torch.zeros(size=[self.S,self.S,self.B*5+self.C])
        objectList=anno['annotation']['object']
        
        boxList=list()

        leftMost=numpy.inf
        topMost=numpy.inf
        rightMost=-1
        botMost=-1
        #temp=numpy.array(img)
        img=tfFunc.to_tensor(img)
        for obj in objectList:
            xmin=int(obj['bndbox']['ymin'])
            ymin=int(obj['bndbox']['xmin'])
            xmax=int(obj['bndbox']['ymax'])
            ymax=int(obj['bndbox']['xmax'])
            cls=obj['name']
            boxList.append([xmin,ymin,xmax,ymax,cls])
            if self.train:
                if ymin<leftMost:
                    leftMost=ymin
                if xmin<topMost:
                    topMost=xmin
                if ymax>rightMost:
                    rightMost=ymax
                if xmax>botMost:
                    botMost=xmax
        if self.train:
            img=self.gray(img)
            img=self.RandomBrightness(img)
            img=self.RandomSaturation(img)
            img=self.RandomHue(img)
            img=self.randomBlur(img)
            cropConstr=[leftMost,topMost,rightMost,botMost]
            img,boxList=self.randomCrop(img,cropConstr,boxList)
        if img.shape!=[3,448,448]:
            img,boxList=self.reSize(img,boxList)
        #cv2.imshow('image',img)

        ####调试使用
        #fig,ax=plt.subplots()
        #ax.imshow(img.permute(1,2,0))

        #####调试end



        #if self.vgg:
            #img=tfFunc.normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        gridSize=self.inputSize//self.S
        for box in boxList:
            y=(box[1]+box[3])//2
            x=(box[0]+box[2])//2
            h=(box[2]-box[0])/self.inputSize
            w=(box[3]-box[1])/self.inputSize
            #调试
            pw=(box[3]-box[1])
            ph=(box[2]-box[0])
            #rec=pch.Rectangle([box[1],box[0]],
            #pw,ph,linewidth=1, edgecolor='r', facecolor='none')
            #ax.add_patch(rec)
            #调试end
            gridX=int(x//gridSize)
            gridY=int(y//gridSize)
            delX=float(x%gridSize)/float(gridSize)
            delY=float(y%gridSize)/float(gridSize)
            if target[gridX,gridY,4]!=1:
                for b in range(self.B):
                    target[gridX,gridY,b*5]=delX
                    target[gridX,gridY,b*5+1]=delY
                    target[gridX,gridY,b*5+2]=h
                    target[gridX,gridY,b*5+3]=w
                    target[gridX,gridY,b*5+4]=1
                target[gridX,gridY,self.B*5+self.classlist[box[4]]]=1
        #plt.savefig('sample.jpg')
        return img,target
    def randomCrop(self,img,constraint,boxs):
        if type(img)!=torch.Tensor:
            img=tfFunc.to_tensor(img)
        imgW=img.shape[2]
        imgH=img.shape[1]
        imgH20=int(imgH*0.2)
        imgW20=int(imgW*0.2)
        imgH80=imgH-imgH20
        imgW80=imgW-imgW20
        top=random.randint(0,constraint[1] if constraint[1]<imgH20 else imgH20)
        left=random.randint(0,constraint[0] if constraint[0]<imgW20 else imgW20)
        width=random.randint(constraint[2] if constraint[2]>imgW80 else imgW80, imgW)-left
        height=random.randint(constraint[3] if constraint[3]>imgH80 else imgH80 ,imgH)-top
        for i,b in enumerate(boxs):
            b[0]-=top
            b[1]-=left
            b[2]-=top
            b[3]-=left
            boxs[i]=b
        return tfFunc.crop(img,top,left,height,width),boxs
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    def RandomBrightness(self,bgr):
        if type(bgr)!=numpy.ndarray:
            bgr=numpy.array(bgr.permute(1,2,0))
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.uniform(0.5,1.5)
            v = v*adjust
            v = numpy.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        bgr=tfFunc.to_tensor(bgr)
        return bgr
    def RandomSaturation(self,bgr):
        if type(bgr)!=numpy.ndarray:
            bgr=numpy.array(bgr.permute(1,2,0))
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.uniform(0.5,1.5)
            s = s*adjust
            s = numpy.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        bgr=tfFunc.to_tensor(bgr)
        return bgr
    def RandomHue(self,bgr):
        if type(bgr)!=numpy.ndarray:
            bgr=numpy.array(bgr.permute(1,2,0))
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = numpy.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        bgr=tfFunc.to_tensor(bgr)
        return bgr
    def randomBlur(self,bgr):
        if type(bgr)!=numpy.ndarray:
            bgr=numpy.array(bgr.permute(1,2,0))
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        bgr=tfFunc.to_tensor(bgr)
        return bgr
    def reSize(self,img,boxes):
        if type(img)!=torch.Tensor:
            img=tfFunc.to_tensor(img)
        imgW=img.shape[2]
        imgH=img.shape[1]
        img=tfFunc.resize(img,[self.inputSize,self.inputSize])
        for i,b in enumerate(boxes):
            boxes[i][0]=b[0]*self.inputSize//imgH
            boxes[i][2]=b[2]*self.inputSize//imgH
            boxes[i][1]=b[1]*self.inputSize//imgW
            boxes[i][3]=b[3]*self.inputSize//imgW
        return img,boxes

        


def loadVOCTrainDataSet(cloud=False,year=2007):
    if year==2007:
        if cloud:
            return datasets.VOCDetection(root=CloudInputDir,
            year='2007',image_set='train',
            transforms=myVOCtransform(VOC_CLASSES,train=True))
        return datasets.VOCDetection(root='../data',
            year='2007',image_set='train',download=False,
            transforms=myVOCtransform(VOC_CLASSES,train=True))
    else:
        if cloud:
            return datasets.VOCDetection(root=CloudInputDir,
            year='2012',image_set='train',
            transforms=myVOCtransform(VOC_CLASSES,train=True))
        return datasets.VOCDetection(root='../data',
            year='2012',image_set='train',download=False,
            transforms=myVOCtransform(VOC_CLASSES,train=True))
def loadVOCValDataSet(cloud=False,year=2007):
    if year==2007:
        if cloud:
            return datasets.VOCDetection(root=CloudInputDir,
            year='2007',image_set='val',
            transforms=myVOCtransform(VOC_CLASSES))
        return datasets.VOCDetection(root='../data',
        year='2007',image_set='val',download=False,
        transforms=myVOCtransform(VOC_CLASSES))
    else:
        if cloud:
            return datasets.VOCDetection(root=CloudInputDir,
            year='2012',image_set='val',
            transforms=myVOCtransform(VOC_CLASSES))
        return datasets.VOCDetection(root='../data',
        year='2012',image_set='val',download=False,
        transforms=myVOCtransform(VOC_CLASSES))

