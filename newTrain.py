import argparse
import numpy
import time
import torch
from myVOCDataSet import loadVOCTrainDataSet, loadVOCValDataSet
from myYOLOModel import pretrainedVGG11
from torch.autograd.variable import Variable
from yoloLoss import yoloLoss
import torch.optim as op
from torch.utils.data import DataLoader
from drawBox import drawBBoxes
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import resize
import cv2



parser = argparse.ArgumentParser(description='PyTorch Yolo Training')
parser.add_argument('--cloud','-c',nargs='?',const=1,default=False,type=bool,help='if on cloud')
parser.add_argument('--gpu','-g',nargs='?',const=1,default=False,type=bool,help='if use gpu')
parser.add_argument('--weighted','-w',nargs='?',const=1,type=str,default=None,help='if use pretrained weight')
parser.add_argument('--batch-size','-b',nargs='?',const=1,default=32,type=int,help='set the batch size')
parser.add_argument('--epochs','-e',nargs='?',const=1,default=50,type=int,help='the epoch number')
parser.add_argument('--num-workers','-n',nargs='?',const=1,default=1,type=int,help='number of workers to load data')
parser.add_argument('--learning-rate','-l',nargs='?',const=1,default=1e-3,type=float)
parser.add_argument('--retrain','-r',nargs='?',const=1,default=None,type=str)
parser.add_argument('--sch-step-size','-s',nargs='?',const=1,default=30,type=int)
parser.add_argument('--download','-dl',nargs='?',const=1,default=False,type=bool)
parser.add_argument('--add-train','-at',nargs='?',const=1,default=False,type=bool)
parser.add_argument('--cloud-input','-ci',nargs='?',const=1,type=str,default='/input0/')
parser.add_argument('--cloud-output','-co',nargs='?',const=1,type=str,default='/output/')
def main():
    args=parser.parse_args()
    print(args)
    cloudOutputDir=args.cloud_output
    cloudInputDir=args.cloud_input
    if args.gpu:
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device='cpu'
    print('using device:'+device)

    writer=SummaryWriter(log_dir='./tf_dir/'+str(time.time()))
    testImg=cv2.imread('test.jpeg')


    model=pretrainedVGG11().to(device)
    if args.weighted is not None:
        model.load_state_dict(torch.load(cloudInputDir+args.weighted))
    if args.retrain is not None:
        model.load_state_dict(torch.load(cloudOutputDir+args.retrain))
    criterion = yoloLoss(7,2,device=device)
    optimizer=op.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=0.0005)
    
    if args.cloud:
        dataRoot=cloudInputDir
    else:
        dataRoot='../data'
    train07DataSet=loadVOCTrainDataSet(root=dataRoot,d=args.download)
    train07Loader=DataLoader(train07DataSet,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    val07DataSet=loadVOCValDataSet(root=dataRoot,d=args.download)
    val07Loader=DataLoader(val07DataSet,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    train12DataSet=loadVOCTrainDataSet(root=dataRoot,year='2012',d=args.download)
    train12Loader=DataLoader(train12DataSet,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)



    
    #sch=op.lr_scheduler.MultiStepLR(optimizer,step_size=args.sch_step_size,verbose=True)
    best_loss=numpy.inf
    print('start training')
    for e in range(args.epochs):
        model.train()
        print('start epoch '+str(e))
        loss=train(model,train07Loader,optimizer,criterion,device)
        if args.add_train:
            loss+=train(model,train12Loader,optimizer,criterion,device)
        writeScalar(writer,'Train',loss[0],loss[1],loss[2],loss[3],loss[4],loss[5])

        model.eval()
        loss=eval(model,val07Loader,criterion,device)
        writeScalar(writer,'Val',loss[0],loss[1],loss[2],loss[3],loss[4],loss[5])
        if best_loss > loss[0]:
            best_loss=loss[0]
            torch.save(model.state_dict(),cloudOutputDir+'bestWeight.pth')
        recordTestImg(model,testImg,224,e,writer)
        #sch.step()
    torch.save(model.state_dict(),cloudOutputDir+'vggYolo.pth')
    writer.close()


def eval(model,loader,lossFunc,device):
    meanJ=0
    meanLoc=0
    meanCon=0
    meanNocon=0
    meanNoobj=0
    meanCls=0
    lenLoader=len(loader)
    for _,(image,target) in enumerate(loader):
        image=Variable(image).to(device)
        target=Variable(target).to(device)
        pred=model(image)
        loss,loc_loss,cont_loss,nocont_loss,noobj_loss,cls_loss=lossFunc(pred,target)
        meanJ+=loss.item()
        meanLoc+=loc_loss.item()
        meanCon+=cont_loss.item()
        meanNocon+=nocont_loss.item()
        meanNoobj+=noobj_loss.item()
        meanCls+=cls_loss.item()
    return torch.tensor([meanJ/lenLoader,meanLoc/lenLoader,meanCon/lenLoader,meanNocon/lenLoader,meanNoobj/lenLoader,meanCls/lenLoader])


def train(model,loader,optimizer,lossFunc,device):
    meanJ=0
    meanLoc=0
    meanCon=0
    meanNocon=0
    meanNoobj=0
    meanCls=0
    lenLoader=len(loader)
    for b,(image,target) in enumerate(loader):
        image=Variable(image).to(device)
        target=Variable(target).to(device)
        pred=model(image,False)
        loss,loc_loss,cont_loss,nocont_loss,noobj_loss,cls_loss=lossFunc(pred,target)
        meanJ+=loss.item()
        meanLoc+=loc_loss.item()
        meanCon+=cont_loss.item()
        meanNocon+=nocont_loss.item()
        meanNoobj+=noobj_loss.item()
        meanCls+=cls_loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.tensor([meanJ/lenLoader,meanLoc/lenLoader,meanCon/lenLoader,meanNocon/lenLoader,meanNoobj/lenLoader,meanCls/lenLoader])

def writeScalar(writer,inWitch,meanJ,meanLoc,meanCon,meanNocon,meanNoobj,meanCls,e):
    writer.add_scalar(inWitch+'loss',meanJ,e)
    writer.add_scalar(inWitch+'Loc',meanLoc,e)
    writer.add_scalar(inWitch+'Cont',meanCon,e)
    writer.add_scalar(inWitch+'Nocont',meanNocon,e)
    writer.add_scalar(inWitch+'Noobj',meanNoobj,e)
    writer.add_scalar(inWitch+'Cls',meanCls,e)

def recordTestImg(model,testImg:numpy.ndarray,inputSize,e,writer):
    model.eval()
    imgTensor=resize(torch.tensor(testImg.transpose([2,0,1])),[inputSize,inputSize])
    pred=model(imgTensor.unsqueeze(0))
    img,_=drawBBoxes(testImg,pred,0.5,7,inputSize)
    writer.add_image('Test/'+str(e), img, e)
if __name__ == '__main__':
    main()