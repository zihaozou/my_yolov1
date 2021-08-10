import argparse
import numpy
import torch
from myVOCDataSet import loadVOCTrainDataSet, loadVOCValDataSet
from myYOLOModel import loadvgg11
from torch.autograd.variable import Variable
from yoloLoss import yoloLoss
import torch.optim as op
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Yolo Training')
parser.add_argument('--cloud','-c',nargs='?',const=1,default=False,type=bool,help='if on cloud')
parser.add_argument('--weighted','-w',nargs='?',const=1,type=str,default=None,help='if use pretrained weight')
parser.add_argument('--batch-size','-b',nargs='?',const=1,default=1,type=int,help='set the batch size')
parser.add_argument('--epochs','-e',nargs='?',const=1,default=50,type=int,help='the epoch number')
parser.add_argument('--num-workers','-n',nargs='?',const=1,default=1,type=int,help='number of workers to load data')
parser.add_argument('--learning-rate','-l',nargs='?',const=1,default=1e-3,type=float)
parser.add_argument('--retrain','-r',nargs='?',const=1,default=False,type=bool)
parser.add_argument('--sch-step-size','-s',nargs='?',const=1,default=30,type=int)
parser.add_argument('--cont-factor','-cf',nargs='?',const=1,default=1.,type=float)
parser.add_argument('--cls-factor','-csf',nargs='?',const=1,default=1.,type=float)
parser.add_argument('--coord-factor','-cof',nargs='?',const=1,default=5.,type=float)
parser.add_argument('--noobj-factor','-nof',nargs='?',const=1,default=.5,type=float)
parser.add_argument('--dropout','-d',nargs='?',const=1,default=.6,type=float)
parser.add_argument('--nc-factor','-ncf',nargs='?',const=1,default=1,type=float)
def main():
    args=parser.parse_args()
    print(args)
    cloud=args.cloud
    if cloud:
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device='cpu'
    print('using device:'+device)
    epochs=args.epochs
    cloudOutputDir='/output/'
    cloudInputDir='/input0/'
    writer=SummaryWriter(log_dir='./tf_dir',)
    batchSize=args.batch_size
    


    model=loadvgg11(cloud=cloud,pretrain=True,do=args.dropout).to(device)
    if args.weighted is not None:
        model.load_state_dict(torch.load(cloudInputDir+args.weighted))
    if args.retrain==True:
        model.load_state_dict(torch.load(cloudOutputDir+'vggYolo.pth'))
    criterion = yoloLoss(7,2,args.coord_factor,args.noobj_factor,cloud=cloud,cf=args.cont_factor,csf=args.cls_factor,nc=args.nc_factor)
    train07DataSet=loadVOCTrainDataSet(cloud=cloud)
    train07Loader=DataLoader(train07DataSet,batch_size=batchSize,shuffle=False,num_workers=args.num_workers)
    val07DataSet=loadVOCValDataSet(cloud=cloud)
    val07Loader=DataLoader(val07DataSet,batch_size=batchSize,shuffle=True,num_workers=args.num_workers)
    train12DataSet=loadVOCTrainDataSet(cloud=cloud,year=2012)
    train12Loader=DataLoader(train12DataSet,batch_size=batchSize,shuffle=True,num_workers=args.num_workers)
    model.train()
    optimizer=op.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=0.0005)
    sch=op.lr_scheduler.StepLR(optimizer,step_size=args.sch_step_size,verbose=True)
    best_loss=numpy.inf
    print('start training')
    for e in range(epochs):
        model.train()
        meanJ=0.
        meanLoc=0.
        meanCon=0.
        meanNocon=0.
        meanNoobj=0.
        meanCls=0.
        print('start epoch '+str(e))
        for b,(image,target) in enumerate(train07Loader):
            image=Variable(image).to(device)
            target=Variable(target).to(device)
            pred=model(image)
            loss,loc_loss,cont_loss,nocont_loss,noobj_loss,cls_loss=criterion(pred,target,val=True)
            if numpy.isnan(loss.item()):
                print('loss diverge')
                exit()
            meanJ+=loss.item()
            meanLoc+=loc_loss.item()
            meanCon+=cont_loss.item()
            meanNocon+=nocont_loss.item()
            meanNoobj+=noobj_loss.item()
            meanCls+=cls_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #trainCnt+=1
            #writer.add_scalar('Loss/trainBatch',loss.item(),trainCnt)
            print(str(e)+': T: set=07, batch='+str(b)+', J='+str(loss.item()))
        #log_file.writelines('epoch '+str(e)+': training mean J ='+str(meanJ/len(trainLoader))+'\n')
        #writer.add_scalar('Loss/train',meanJ/len(trainLoader),e)
        for b,(image,target) in enumerate(train12Loader):
            image=Variable(image).to(device)
            target=Variable(target).to(device)
            pred=model(image)
            loss,loc_loss,cont_loss,nocont_loss,noobj_loss,cls_loss=criterion(pred,target,val=True)
            if numpy.isnan(loss.item()):
                print('loss diverge')
                exit()
            meanJ+=loss.item()
            meanLoc+=loc_loss.item()
            meanCon+=cont_loss.item()
            meanNocon+=nocont_loss.item()
            meanNoobj+=noobj_loss.item()
            meanCls+=cls_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #trainCnt+=1
            #writer.add_scalar('Loss/trainBatch',loss.item(),trainCnt)
            print(str(e)+': T: set=12, batch='+str(b)+', J='+str(loss.item()))
        writer.add_scalar('Loss/train',meanJ/(len(train07Loader)+len(train12Loader)),e)
        writer.add_scalar('Loss/trainLoc',meanLoc/(len(train07Loader)+len(train12Loader)),e)
        writer.add_scalar('Loss/trainCont',meanCon/(len(train07Loader)+len(train12Loader)),e)
        writer.add_scalar('Loss/trainNocont',meanNocon/(len(train07Loader)+len(train12Loader)),e)
        writer.add_scalar('Loss/trainNoobj',meanNoobj/(len(train07Loader)+len(train12Loader)),e)
        writer.add_scalar('Loss/trainCls',meanCls/(len(train07Loader)+len(train12Loader)),e)
        model.eval()
        meanJ=0.
        meanLoc=0.
        meanCon=0.
        meanNocon=0.
        meanNoobj=0.
        meanCls=0.
        for v,(image,target) in enumerate(val07Loader):
            image=Variable(image).to(device)
            target=Variable(target).to(device)
            pred=model(image)
            loss,loc_loss,cont_loss,nocont_loss,noobj_loss,cls_loss=criterion(pred,target,True)
            meanJ+=loss.item()
            meanLoc+=loc_loss.item()
            meanCon+=cont_loss.item()
            meanNocon+=nocont_loss.item()
            meanNoobj+=noobj_loss.item()
            meanCls+=cls_loss.item()
            #valCnt+=1
            #writer.add_scalar('Loss/valBatch',loss.item(),valCnt)
            print(str(e)+': V: set=07, batch='+str(v)+', J='+str(loss.item()))
        writer.add_scalar('Loss/valLoc',meanLoc/len(val07Loader),e)
        writer.add_scalar('Loss/valCont',meanCon/len(val07Loader),e)
        writer.add_scalar('Loss/valNocont',meanNocon/len(val07Loader),e)
        writer.add_scalar('Loss/valNoobj',meanNoobj/len(val07Loader),e)
        writer.add_scalar('Loss/valCls',meanCls/len(val07Loader),e)
            
        #log_file.writelines('epoch '+str(e)+': validation mean J ='+str(meanJ)+'\n')
        writer.add_scalar('Loss/val',meanJ/len(val07Loader),e)
        if best_loss > meanJ/len(val07Loader):
            best_loss=meanJ/len(val07Loader)
            torch.save(model.state_dict(),cloudOutputDir+'bestWeight.pth')
        #print('J='+str(meanJ/batchCnt))
        #log_file.flush()
        sch.step()
    torch.save(model.state_dict(),cloudOutputDir+'vggYolo.pth')
    writer.close()


if __name__ == '__main__':
    main()