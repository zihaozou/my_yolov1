from drawBox import drawBBoxes
import random
import torch
from torchvision import datasets
import torchvision.transforms.functional as tfFunc
from PIL import Image, ImageEnhance, ImageFilter
from PIL import Image,ImageEnhance,ImageFilter
CloudInputDir='/input0'
VOC_CLASSES = {    # always index 0
    'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3,
    'bottle':4, 'bus':5, 'car':6, 'cat':7, 'chair':8,
    'cow':9, 'diningtable':10, 'dog':11, 'horse':12,
    'motorbike':13, 'person':14, 'pottedplant':15,
    'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class myVOCtransform(object):
    changeFactor=[0.6,1,1.2,1.3,1.5,1.8]
    def __init__(self,classlist,S=7,B=2,C=20,inputSize=224,train=False):
        super().__init__()
        self.S=S
        self.B=B
        self.C=C
        self.inputSize=inputSize
        self.classlist=classlist
        self.train=train
    def __call__(self, img,anno):
        random.seed()
        objectList=anno['annotation']['object']
        boxList=torch.FloatTensor(size=[len(objectList),5])
        for i,obj in enumerate(objectList):#妈的voc的index从1开始算
            xmin=int(obj['bndbox']['ymin'])-1
            ymin=int(obj['bndbox']['xmin'])-1
            xmax=int(obj['bndbox']['ymax'])-1
            ymax=int(obj['bndbox']['xmax'])-1
            cls=obj['name']
            boxList[i,:]=torch.tensor([xmin,ymin,xmax,ymax,VOC_CLASSES[cls]+self.B*5])
        target=torch.zeros(size=[self.S,self.S,self.B*5+self.C])
        #originImg=img
        if self.train:
            img=self.randomBrightness(img)
            img=self.randomSharporBlur(img)
            img=self.randomContrast(img)
            img=self.randomSatuation(img)
            img,boxList=self.randomFlip(img,boxList)
            #img,boxList=self.randomCrop(img,boxList)#random crop因效果很差被取消
        if img.size!=(3,self.inputSize,self.inputSize):
            img,boxList=self.reSize(img,boxList)
        gridSize=self.inputSize//self.S
        for _,box in enumerate(boxList):
            y=(box[1]+box[3])/2
            x=(box[0]+box[2])/2
            h=(box[2]-box[0])/self.inputSize
            w=(box[3]-box[1])/self.inputSize
            gridX=int(x//gridSize)
            gridY=int(y//gridSize)
            if gridX>=7 or gridY>=7:
                print(gridX,gridY)
            delX=float(x%gridSize)/float(gridSize)
            delY=float(y%gridSize)/float(gridSize)
            if target[gridX,gridY,4]!=1:
                for b in range(self.B):
                    target[gridX,gridY,b*5]=delX
                    target[gridX,gridY,b*5+1]=delY
                    target[gridX,gridY,b*5+2]=h
                    target[gridX,gridY,b*5+3]=w
                    target[gridX,gridY,b*5+4]=1
                target[gridX,gridY,int(box[4])]=1
        #imgArr=numpy.array(originImg)
        #imgSave,_=drawBBoxes(imgArr,target,0.5,32,224,'cpu')
        #tfFunc.to_pil_image(imgSave).save('save.jpg')
        return tfFunc.to_tensor(img),target
    def randomFlip(self,img:Image.Image,boxes):
        if random.random()<.5:
            W,_=img.size
            img=img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes[:,[1,3]]=W-boxes[:,[3,1]]-1
        return img,boxes
    def randomCrop(self,img:Image.Image,boxes:torch.Tensor):
        #只有剪切后剩余面积大于原面积的60%的bbox才会被保留
        if random.random()>=0:
            W,H=img.size
            N=boxes.size(0)
            left=random.randint(0,int(W*.2))
            top=random.randint(0,int(H*.2))
            right=random.randint(int(W*.8),W)
            bot=random.randint(int(H*.8),H)
            cropBox=torch.tensor([left,top,right,bot]).unsqueeze(0)
            lt = torch.max(#计算左上角盒子的顶点
                boxes[:,:2].unsqueeze(1).expand(N,1,2),
                cropBox[:,:2].unsqueeze(0).expand(N,1,2),
            )
            rb = torch.min(#计算右下角盒子的顶点
                boxes[:,2:4].unsqueeze(1).expand(N,1,2),
                cropBox[:,2:].unsqueeze(0).expand(N,1,2),
            )
            wh = rb - lt
            wh[wh<0] = 0  # clip at 0
            inter = (wh[:,:,0] * wh[:,:,1]).squeeze()
            boxesArea=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            mask=((inter/boxesArea)>=.6).unsqueeze(-1).expand_as(boxes)
            boxes=boxes[mask].view(-1,5)
            boxes[:,:2]=boxes[:,:2]-cropBox[:,:2].expand_as(boxes[:,:2])
            boxes[:,2:4]=boxes[:,2:4]-cropBox[:,:2].expand_as(boxes[:,2:4])
            boxes[boxes[:,0]<0,0]=0
            boxes[boxes[:,1]<0,1]=0
            boxes[boxes[:,2]>(bot-top),2]=bot-top-1
            boxes[boxes[:,3]>(right-left),3]=right-left-1
            img=img.crop((left,top,right,bot))
        return img,boxes
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
    def reSize(self,img:Image.Image,boxes):
        imgW,imgH=img.size
        img=img.resize((self.inputSize,self.inputSize))
        resizeFactor=torch.tensor([self.inputSize/imgH,self.inputSize/imgW,self.inputSize/imgH,self.inputSize/imgW]).unsqueeze(0).expand_as(boxes[:,:4])
        boxes[:,:4]=boxes[:,:4]*resizeFactor
        return img,boxes

        


def loadVOCTrainDataSet(root='../root',year='2007',d=False):
    return datasets.VOCDetection(root=root,
        year=year,image_set='train',download=d,
        transforms=myVOCtransform(VOC_CLASSES,train=True))
def loadVOCValDataSet(root='../root',year='2007',d=False):
    return datasets.VOCDetection(root=root,
    year=year,image_set='val',download=d,
    transforms=myVOCtransform(VOC_CLASSES))


