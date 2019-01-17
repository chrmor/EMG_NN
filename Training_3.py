
# coding: utf-8

# In[1]:

##Import libraries
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import time
import datetime
import os
import sys
import csv
from torch import backends
from beautifultable import BeautifulTable
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# In[53]:

##SETTINGS

nfold = 3 #number of folds to train
lr=0.1 #learning rate

batch_size = 32
val_split = .1 #trainset percentage allocated for devset
test_val_split = .1 #trainset percentage allocated for test_val set (i.e. the test set of known patients)


spw=20 #samples per window
nmuscles=10 #initial number of muscles acquired

#Enable/Disable shuffle on trainset/testset
shuffle_train = True 
shuffle_test= True

#Delete electrogonio signals
exclude_features=True
#Only use electrogonio signals
include_only_features=False
#Features to selected/deselected for input to the networks
features_select = [9,10] #1 to 4

#Select which models to run. Insert comma separated values into 'model_select' var.
#List. 0:'FF', 1:'FC2', 2:'FC2DP', 3:'FC3', 4:'FC3dp', 5:'Conv1d', 6:'MultiConv1d' 
#e.g: model_select = [0,4,6] to select FF,FC3dp,MultiConv1d
model_lst = ['FF','FC2','FC2DP','FC3','FC3dp','Conv1d','MultiConv1d',
             'MultiConv1d_2','MultiConv1d_3', 'MultiConv1d_4', 'MultiConv1d_5', 'FF2', 'CNN1', 'FF3', 'FF4', 'CNN2', 'FF5', 'FF6']
model_select = [16,17] 

#Early stop settings
maxepoch = 100
maxpatience = 10

use_cuda = True
use_gputil = True
cuda_device = None


# In[54]:

#CUDA

if use_gputil and torch.cuda.is_available():
    import GPUtil

    # Get the first available GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=100, maxMemory=20)  # return a list of available gpus
    except:
        print('GPU not compatible with NVIDIA-SMI')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
        
    ttens = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    ttens = ttens.cuda()
    


# In[55]:

#torch.cuda.is_available()


# In[56]:

#Seeds
def setSeeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
setSeeds(0)


# In[57]:

#Prints header of beautifultable report for each fold
def header(model_list,nmodel,nfold,traindataset,testdataset):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('MODEL: '+model_list[nmodel])
    print('Fold: '+str(nfold))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
    shape = list(traindataset.x_data.shape)
    print('Trainset fold'+str(i)+' shape: '+str(shape[0])+'x'+str((shape[1]+1)))
    shape = list(testdataset.x_data.shape)
    print('Testset fold'+str(i)+' shape: '+str(shape[0])+'x'+str((shape[1]+1))+'\n')


# In[58]:

#Prints actual beautifultable for each fold
def table(model_list,nmodel,accuracies,precisions,recalls,f1_scores,accuracies_dev):
    table = BeautifulTable()
    table.column_headers = ["{}".format(model_list[nmodel]), "Avg", "Stdev"]
    table.append_row(["Accuracy",round(np.average(accuracies),3),round(np.std(accuracies),3)])
    table.append_row(["Precision",round(np.average(precisions),3),round(np.std(precisions),3)])
    table.append_row(["Recall",round(np.average(recalls),3),round(np.std(recalls),3)])
    table.append_row(["F1_score",round(np.average(f1_scores),3),round(np.std(f1_scores),3)])
    table.append_row(["Accuracy_dev",round(np.average(accuracies_dev),3),round(np.std(accuracies_dev),3)])    
    print(table)


# In[59]:

#Saves best model state on disk for each fold
def save_checkpoint (state, is_best, filename, logfile):
    if is_best:
        msg = "=> Saving a new best. "+'Epoch: '+str(state['epoch'])
        print (msg)
        logfile.write(msg + "\n")
        torch.save(state, filename)  
    else:
        msg = "=> Validation accuracy did not improve. "+'Epoch: '+str(state['epoch'])
        print (msg)
        logfile.write(msg + "\n")


# In[60]:

#Compute sklearn metrics: Recall, Precision, F1-score
def pre_rec (loader, model):
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for i,data in enumerate (loader,0):
            inputs, labels = data
            y_true = np.append(y_true,labels)
            outputs = model(inputs)
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            y_pred = np.append(y_pred,outputs)
    y_true = np.where(y_true==0.0,0,1)
    y_pred = np.where(y_pred==0.0,0,1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return round(precision,3), round(recall,3), round(f1_score,3)


# In[61]:

#Calculates model accuracy. Predicted vs Correct.
def accuracy (loader, model):
    total=0
    correct=0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    return round((100 * correct / total),3)


# In[62]:

#Arrays to store metrics
accs = np.empty([nfold,1])
accs_test_val = np.empty([nfold,1])
precisions = np.empty([nfold,1])
recalls = np.empty([nfold,1])
f1_scores = np.empty([nfold,1])
accs_dev = np.empty([nfold,1])
times = np.empty([nfold,1])

#Calculate avg metrics on folds
def averages (accs,precs,recs,f1,accs_dev):
    a = round(np.average(accs),3)
    p = round(np.average(precs),3)
    r = round(np.average(recs),3)
    f = round(np.average(f1),3)
    a_d = round(np.average(accs_dev),3)
    return a,p,r,f,a_d

#Calculate std metrics on folds
def stds (accs,precs,recs,f1,accs_dev):
    a = round(np.std(accs),3)
    p = round(np.std(precs),3)
    r = round(np.std(recs),3)
    f = round(np.std(f1),3)
    a_d = round(np.std(accs_dev),3)
    return a,p,r,f,a_d


# In[63]:

#Shuffle
def dev_shuffle (shuffle_train,shuffle_test,val_split,traindataset,testdataset):
    train_size = len(traindataset)
    test_size = len(testdataset)
    train_indices = list(range(train_size))
    test_indices = list(range(test_size))
    split = int(np.floor(val_split * train_size))
    if shuffle_train:
        np.random.shuffle(train_indices)
    if shuffle_test:
        np.random.shuffle(test_indices) 
    train_indices, dev_indices = train_indices[split:], train_indices[:split]
    # Samplers
    tr_sampler = SubsetRandomSampler(train_indices)
    d_sampler = SubsetRandomSampler(dev_indices)
    te_sampler = SubsetRandomSampler(test_indices)
    return tr_sampler,d_sampler,te_sampler

def data_split (shuffle_train,shuffle_test,val_split,test_val_split,traindataset,testdataset):
    train_size = len(traindataset)
    test_size = len(testdataset)
    train_indices = list(range(train_size))
    test_indices = list(range(test_size))
    test_val_split = int(np.floor(test_val_split * train_size)) 
    dev_split = int(np.floor(val_split * (train_size-test_val_split) + test_val_split))
    if shuffle_train:
        np.random.shuffle(train_indices)
    if shuffle_test:
        np.random.shuffle(test_indices) 
    train_indices, dev_indices, test_val_indices = train_indices[dev_split:], train_indices[test_val_split:dev_split], train_indices[:test_val_split]
    # Samplers
    tr_sampler = SubsetRandomSampler(train_indices)
    d_sampler = SubsetRandomSampler(dev_indices)
    tv_sampler = SubsetRandomSampler(test_val_indices)                
    te_sampler = SubsetRandomSampler(test_indices)
    return tr_sampler,d_sampler,tv_sampler,te_sampler


# In[64]:

'''
test_val_split = 0.1
train_size = 100
val_split = 0.1
test_val_split = int(np.floor(test_val_split * train_size)) 
dev_split = int(np.floor(val_split * (train_size-test_val_split) + test_val_split))
print(str(test_val_split) + " " + str(dev_split))

train_indices = []
for i in range(0,100):
    train_indices.append(i + 1)

print(train_indices)
    
train_indices, dev_indices, test_val_indices = train_indices[dev_split:], train_indices[test_val_split:dev_split], train_indices[:test_val_split]

print("Train: " + str(train_indices))
print("Test Val: " + str(test_val_indices))
print("Dev: " + str(dev_indices))
'''


# In[65]:

#Loads and appends all folds all at once
trainfolds = []
testfolds = []
cwd = os.getcwd()
#l=pd.read_csv(cwd +'/list.csv',sep=',',header=None,dtype=np.int32)
col_select = np.array([])


for i in range (spw*nmuscles,200):
    col_select = np.append(col_select,i)
    
for i in range (0,spw*nmuscles,nmuscles):
    for muscle in features_select:
        col_select = np.append(col_select,muscle -1 + i)
    cols=np.arange(0,201)

if exclude_features & (not include_only_features): #delete gonio
    for j in range(1,nfold+1):
        print("Loading fold " + str(j))
        traindata = pd.read_table(os.path.join(cwd,'TrainFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32,usecols=[i for i in cols if i not in col_select.astype(int)])
        testdata = pd.read_table(os.path.join(cwd,'TestFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i not in col_select.astype(int)])
        trainfolds.append(traindata)
        testfolds.append(testdata) 
elif include_only_features & (not exclude_features): #only gonio
    for j in range(1,nfold+1):
        print("Loading fold " + str(j))
        traindata = pd.read_table(os.path.join(cwd,'TrainFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32,usecols=[i for i in cols if i in col_select.astype(int)])
        testdata = pd.read_table(os.path.join(cwd,'TestFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i in col_select.astype(int)])
        trainfolds.append(traindata)
        testfolds.append(testdata) 
elif (not include_only_features) & (not exclude_features): 
    for j in range(1,nfold+1):
        print("Loading fold " + str(j))
        traindata = pd.read_csv(os.path.join(cwd,'TrainFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32)
        testdata = pd.read_csv(os.path.join(cwd,'TestFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32)
        trainfolds.append(traindata)
        testfolds.append(testdata)
else:
    raise ValueError('use_gonio and del_gonio cannot be both True')

nmuscles=int((len(traindata.columns)-1)/spw) #used for layer dimensions and stride CNNs
print(len(traindata.columns))
print(nmuscles)


# In[66]:

import models
from models import *
models._spw = spw
models._nmuscles = nmuscles
models._batch_size = batch_size


# In[67]:

print(models._nmuscles)

#import models
#from models import *
#TEST DIMENSIONS
#models.nmuscles = nmuscles
def testdimensions():
    model = Model3()
    print(model)
    x = torch.randn(32,1,160)
    #model.test_dim(x)
 
testdimensions()


# In[68]:

fieldnames = ['Fold','Acc_test_val', 'Accuracy','Precision','Recall','F1_score','Stop_epoch','Accuracy_dev'] #coloumn names report FOLD CSV
torch.backends.cudnn.benchmark = True

#TRAINING LOOP
def train_test():
    for k in model_select:
        
        table = BeautifulTable()
        avgtable = BeautifulTable()
        fieldnames1 = [model_lst[k],'Avg','Std_dev'] #column names report GLOBAL CSV
        folder = os.path.join(cwd,'Report_'+str(model_lst[k]))
        if not os.path.exists(folder):
            os.mkdir(folder)

        logfilepath = os.path.join(folder,'log.txt')
        logfile = open(logfilepath,"w") 

        with open(os.path.join(folder,'Report_folds.csv'),'w') as f_fold, open(os.path.join(folder,'Report_global.csv'),'w') as f_global:
            writer = csv.DictWriter(f_fold, fieldnames = fieldnames)
            writer1  = csv.DictWriter(f_global, fieldnames = fieldnames1)
            writer.writeheader()
            writer1.writeheader()
            t0 = 0
            t1 = 0
            for i in range(1,nfold+1):
                t0 = time.time()
                
                setSeeds(0)
                
                class Traindataset(Dataset):
                    def __init__(self):
                        self.data=trainfolds[i-1]
                        self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1])) 
                        self.len=self.data.shape[0]
                        self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]]))
                        if (use_cuda):
                            self.x_data = self.x_data.cuda()
                            self.y_data = self.y_data.cuda()
                    def __getitem__(self, index):
                        return self.x_data[index], self.y_data[index]
                    def __len__(self):
                        return self.len
                class Testdataset(Dataset):
                    def __init__(self):
                        self.data=testfolds[i-1]
                        self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1]))
                        self.len=self.data.shape[0]
                        self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]]))
                        if (use_cuda):
                            self.x_data = self.x_data.cuda()
                            self.y_data = self.y_data.cuda()
                    def __getitem__(self, index):
                        return self.x_data[index], self.y_data[index]
                    def __len__(self):
                        return self.len

                traindataset = Traindataset()
                testdataset = Testdataset()

                header(model_lst,k,i,traindataset,testdataset)

                #train_sampler,dev_sampler,test_sampler=dev_shuffle(shuffle_train,shuffle_test,val_split,traindataset,testdataset)
                train_sampler,dev_sampler,test_val_sampler,test_sampler=data_split(shuffle_train,shuffle_test,val_split,test_val_split,traindataset,testdataset)
                #loaders
                train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, 
                                                           sampler=train_sampler,drop_last=True)
                test_val_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size,
                                                                sampler=test_val_sampler,drop_last=True)
                dev_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, 
                                                           sampler=dev_sampler,drop_last=True)
                test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                                                sampler=test_sampler,drop_last=True)

                if k==0:
                    model=Model0()
                if k==1:
                    model=Model1()
                if k==2:
                    model=Model2()
                if k==3:
                    model=Model3()
                if k==4:
                    model=Model4()
                if k==5:
                    model=Model5()
                if k==6:
                    model=Model6()
                if k==7:
                    model=Model7()
                if k==8:
                    model=Model8()
                if k==9:
                    model=Model9()
                if k==10:
                    model=Model10()
                if k==11:
                    model=Model11()  
                if k==12:
                    model=Model12()
                if k==13:
                    model=Model13()    
                if k==14:
                    model=Model14() 
                if k==15:
                    model=Model15() 
                if k==16:
                    model=Model16() 
                if k==17:
                    model=Model17()                     
                if (use_cuda):
                    model = model.cuda()

                criterion = nn.BCELoss(size_average=True)
                optimizer = torch.optim.SGD(model.parameters(), lr)    
                msg = 'Accuracy on test set before training: '+str(accuracy(test_loader, model))+'\n'
                print(msg)
                logfile.write(msg + "\n")
                #EARLY STOP
                epoch = 0
                patience = 0
                best_acc_dev=0
                while (epoch<maxepoch and patience < maxpatience):
                    running_loss = 0.0
                    for l, data in enumerate(train_loader, 0):
                        inputs, labels = data
                        if use_cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        inputs, labels = Variable(inputs), Variable(labels)
                        y_pred = model(inputs)
                        if use_cuda:
                            y_pred = y_pred.cuda()
                        loss = criterion(y_pred, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        #print accuracy ever l mini-batches
                        if l % 2000 == 1999:
                            msg = '[%d, %5d] loss: %.3f' %(epoch + 1, l + 1, running_loss / 999)
                            print(msg)
                            logfile.write(msg + "\n")
                            running_loss = 0.0
                            #msg = 'Accuracy on dev set:' + str(accuracy(dev_loader))
                            #print(msg)
                            #logfile.write(msg + "\n")        
                    accdev = (accuracy(dev_loader, model))
                    msg = 'Accuracy on dev set:' + str(accdev)
                    print(msg)
                    logfile.write(msg + "\n")        
                    is_best = bool(accdev > best_acc_dev)
                    best_acc_dev = (max(accdev, best_acc_dev))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc_dev': best_acc_dev
                    }, is_best,os.path.join(folder,'F'+str(i)+'best.pth.tar'), logfile)
                    if is_best:
                        patience=0
                    else:
                        patience = patience+1
                    epoch = epoch+1
                    logfile.flush()
                state = torch.load(os.path.join(folder,'F'+str(i)+'best.pth.tar'))
                stop_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                accuracy_dev = state['best_acc_dev']
                model.eval()
                acctest = (accuracy(test_loader, model))
                acctest_val = (accuracy(test_val_loader, model))
                accs[i-1] = acctest
                accs_test_val[i-1] = acctest_val
                precision,recall,f1_score = pre_rec(test_loader, model)
                precisions[i-1] = precision
                recalls[i-1] = recall
                f1_scores[i-1] = f1_score
                accs_dev[i-1] = accuracy_dev
                writer.writerow({'Fold': i,'Acc_test_val': acctest_val, 'Accuracy': acctest,'Precision': precision,'Recall': recall,'F1_score': f1_score,'Stop_epoch': stop_epoch,'Accuracy_dev': accuracy_dev})
                table.column_headers = fieldnames
                table.append_row([i,acctest_val,acctest,precision,recall,f1_score,stop_epoch,accuracy_dev])
                print(table)
                print('----------------------------------------------------------------------')
                logfile.write(str(table) + "\n----------------------------------------------------------------------\n")
                t1 = time.time()
                times[i-1] = int(t1-t0)
            duration = str(datetime.timedelta(seconds=np.sum(times)))
            writer.writerow({})
            writer.writerow({'Fold': 'Elapsed time: '+duration})
            avg_acc_test_val = round(np.average(accs_test_val),3)
            std_acc_test_val = round(np.std(accs_test_val),3)
            avg_a,avg_p,avg_r,avg_f,avg_a_d=averages(accs,precisions,recalls,f1_scores,accs_dev)
            std_a,std_p,std_r,std_f,std_a_d=stds(accs,precisions,recalls,f1_scores,accs_dev)
            writer1.writerow({model_lst[k]: 'Accuracy','Avg': avg_a,'Std_dev': std_acc_test_val})
            writer1.writerow({model_lst[k]: 'Accuracy test val','Avg': avg_acc_test_val,'Std_dev': std_a})
            writer1.writerow({model_lst[k]: 'Precision','Avg': avg_p,'Std_dev': std_p})
            writer1.writerow({model_lst[k]: 'Recall','Avg': avg_r,'Std_dev': std_r})
            writer1.writerow({model_lst[k]: 'F1_score','Avg': avg_f,'Std_dev': std_f})
            writer1.writerow({model_lst[k]: 'Accuracy_dev','Avg': avg_a_d,'Std_dev': std_a_d})
            writer1.writerow({})
            writer1.writerow({model_lst[k]: 'Elapsed time: '+duration})
            avgtable.column_headers = fieldnames1
            avgtable.append_row(['Accuracy',avg_a,std_a])
            avgtable.append_row(['Accuracy test val',avg_acc_test_val,std_acc_test_val])
            avgtable.append_row(['Precision',avg_p,std_p])
            avgtable.append_row(['Recall',avg_r,std_r])
            avgtable.append_row(['F1_score',avg_a,std_f])
            avgtable.append_row(['Accuracy_dev',avg_a_d,std_a_d])
            print(avgtable)
            logfile.write(str(avgtable) + "\n")
            msg = 'Elapsed time: '+ duration + '\n\n'
            print(msg)
            logfile.write(msg )

        logfile.close()
        


# In[69]:

nmuscles=int((len(traindata.columns)-1)/spw)
if use_cuda and not use_gputil and cuda_device!=None and torch.cuda.is_available():
    with torch.cuda.device(cuda_device):
        train_test()
else:
    train_test()


# In[ ]:




# In[ ]:



