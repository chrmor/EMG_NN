
# coding: utf-8

# In[3]:

##SETTINGS

nfold = 10 #number of folds to train
lr=0.1 #learning rate

batch_size = 32
val_split = .1 #trainset percentage allocated for devset
spw=20 #samples per window
nmuscles=10 #initial number of muscles acquired

#Enable/Disable shuffle on trainset/testset
shuffle_train = True 
shuffle_test= True

#Delete electrogonio signals
del_gonio=True

#Only use electrogonio signals
use_gonio=False

#Select which models to run. Insert comma separated values into 'model_select' var.
#List. 0:'FF', 1:'FC2', 2:'FC2DP', 3:'FC3', 4:'FC3dp', 5:'Conv1d', 6:'MultiConv1d' 
#e.g: model_select = [0,4,6] to select FF,FC3dp,MultiConv1d
model_lst = ['FF','FC2','FC2DP','FC3','FC3dp','Conv1d','MultiConv1d']
model_select = [3,6] 

#Early stop settings
maxepoch = 100
maxpatience = 5


# In[4]:

##Import libraries
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
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


# In[5]:

#CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[6]:

torch.cuda.is_available()


# In[7]:

#Seeds
torch.manual_seed(5)
random.seed(10)
np.random.seed(20)


# In[8]:

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


# In[9]:

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


# In[10]:

#Saves best model state on disk for each fold
def save_checkpoint (state, is_best, filename):
    if is_best:
        print ("=> Saving a new best. "+'Epoch: '+str(state['epoch']))
        torch.save(state, filename)  
    else:
        print ("=> Validation accuracy did not improve. "+'Epoch: '+str(state['epoch']))


# In[11]:

#Compute sklearn metrics: Recall, Precision, F1-score
def pre_rec (loader):
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


# In[12]:

#Calculates model accuracy. Predicted vs Correct.
def accuracy (loader):
    total=0
    correct=0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            outputs = model(inputs.to(device))
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    return round((100 * correct / total),3)


# In[13]:

#Arrays to store metrics
accs = np.empty([nfold,1])
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


# In[14]:

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


# In[15]:

#Loads and appends all folds all at once
trainfolds = []
testfolds = []
cwd = os.getcwd()
#l=pd.read_csv(cwd +'/list.csv',sep=',',header=None,dtype=np.int32)
col_del = np.array([])
for i in range (8,200,nmuscles):
    col_del = np.append(col_del,i)
    col_del = np.append(col_del,i+1)
    cols=np.arange(0,201)
col_only = np.append(col_del,200)
if del_gonio & (not use_gonio): #delete gonio
    for j in range(1,nfold+1):
        traindata = pd.read_table((cwd+'/TrainFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32,usecols=[i for i in cols if i not in col_del.astype(int)])
        testdata = pd.read_table((cwd+'/TestFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i not in col_del.astype(int)])
        trainfolds.append(traindata)
        testfolds.append(testdata) 
elif use_gonio & (not del_gonio): #only gonio
    for j in range(1,nfold+1):
        traindata = pd.read_table((cwd+'/TrainFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32,usecols=[i for i in cols if i in col_only.astype(int)])
        testdata = pd.read_table((cwd+'/TestFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i in col_only.astype(int)])
        trainfolds.append(traindata)
        testfolds.append(testdata) 
elif (not use_gonio) & (not del_gonio): 
    for j in range(1,nfold+1):
        traindata = pd.read_csv((cwd+'/TrainFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32)
        testdata = pd.read_csv((cwd+'/TestFold'+str(j)+'.csv'),sep=',',header=None,dtype=np.float32)
        trainfolds.append(traindata)
        testfolds.append(testdata)
else:
    raise ValueError('use_gonio and del_gonio cannot be both True')

nmuscles=int((len(traindata.columns)-1)/20) #used for layer dimensions and stride CNNs


# In[16]:

trainfolds[0]


# In[21]:

#List of all models. Common activation function: ReLu. Common dp_ratio=0.5. Last activation function: sigmoid.

#1 hidden layer
class Model0(torch.nn.Module):
    def __init__(self):
        super(Model0,self).__init__()
        self.l1 = torch.nn.Linear(spw*nmuscles,32)
        self.l2 = torch.nn.Linear(32,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        y_pred=self.sigmoid(self.l2(out))
        return y_pred


#2 hidden layers
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1,self).__init__()
        self.l1 = torch.nn.Linear(spw*nmuscles,64)
        self.l2 = torch.nn.Linear(64,32)
        self.l3 = torch.nn.Linear(32,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        y_pred = self.sigmoid(self.l3(out))
        return y_pred

#2 hidden layers w/dropout
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2,self).__init__()
        self.l1 = torch.nn.Linear(spw*nmuscles,64)
        self.l1_dropout = nn.Dropout(p = 0.5)
        self.l2 = torch.nn.Linear(64,32)
        self.l2_dropout = nn.Dropout(p = 0.5)
        self.l3 = torch.nn.Linear(32,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        out = F.relu(self.l1_dropout(self.l1(x)))
        out = F.relu(self.l2_dropout(self.l2(out)))
        y_pred = self.sigmoid(self.l3(out))
        return y_pred

#3 hidden layers
class Model3(torch.nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.l1 = torch.nn.Linear(spw*nmuscles, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, 128)
        self.l4 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        y_pred = self.sigmoid(self.l4(out))
        return y_pred

#3 hidden layers w/dropout
class Model4(torch.nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.l1 = torch.nn.Linear(spw*nmuscles, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l2_dropout = nn.Dropout(p=0.5)
        self.l3 = torch.nn.Linear(512, 128)
        self.l3_dropout = nn.Dropout(p=0.5)
        self.l4 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2_dropout(self.l2(out)))
        out = F.relu(self.l3_dropout(self.l3(out)))
        y_pred = self.sigmoid(self.l4(out))
        return y_pred

#1D convnet w/o dropout
class Model5(nn.Module):
    def __init__(self):
        super(Model5,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=(5*nmuscles),stride=nmuscles)
        self.mp = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(8,128)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)
    def forward(self,x):
        x = x.view(batch_size,1,-1)
        #print(x.shape)
        x = F.relu(self.mp(self.conv1(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        y_pred = self.fc3(x)
        return F.sigmoid(y_pred)

#1D Convnet -> Stride=nmuscles. Multi Filter:2xnmuscles,4x,6x,8x. Multi channel:20.
class Model6(nn.Module):
    def __init__(self,**kwargs):
        super(Model6,self).__init__()
        self.FILTERS=[5*nmuscles,10*nmuscles]
        conv_out_channels = 20 
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.MaxPool1d(2)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(260,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,1)

        
    def forward(self,x):
        x = x.view(batch_size,1,-1)
        #print("INPUT SIZE: " + str(x.shape))
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x= x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))


# In[23]:

#TEST DIMENSIONS
def testdimensions():
    model = Model6()

    for f in model.FILTERS:
        print("Filter size: " + str(f))

    x = torch.randn(32,1,160)
    print("Input:" + str(x.shape))
    print("Features: " + str(nmuscles))
    print("Do convnets")
    x = [F.relu(conv(x)) for conv in model.convs]
    for out in x:
        print("Out: " + str(out.shape))
    print("Do maxpool")
    x = [model.mp(i) for i in x]   
    for i in x:
        print("Out: " + str(i.shape))
    print("Do concat")    
    x = torch.cat(x,2)
    x = x.view(32,1,-1).squeeze()
    print("Out: " + str(x.shape))

    print("Do Fully connected 1")
    x = model.fc1(x)
    print("Out: " + str(x.shape))
    print("Do Fully connected 2")
    x = model.fc2(x)
    print("Out: " + str(x.shape))
    print("Do Fully connected 3")
    x = model.fc3(x)
    print("Out: " + str(x.shape))
    print("Do Fully connected 4")
    x = model.fc4(x)
    print("Out: " + str(x.shape))
    
#testdimensions()


# In[ ]:

fieldnames = ['Fold','Accuracy','Precision','Recall','F1_score','Stop_epoch','Accuracy_dev'] #coloumn names report FOLD CSV
torch.backends.cudnn.benchmark = True

#TRAINING LOOP
for k in model_select:
    table = BeautifulTable()
    avgtable = BeautifulTable()
    fieldnames1 = [model_lst[k],'Avg','Std_dev'] #column names report GLOBAL CSV
    folder = cwd+'/Report_'+str(model_lst[k])
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    logfilepath = folder + '/log.txt'
    logfile = open(logfilepath,"w") 
    
    with open(folder+'/Report_folds.csv','w') as f_fold, open(folder+'/Report_global.csv','w') as f_global:
        writer = csv.DictWriter(f_fold, fieldnames = fieldnames)
        writer1  = csv.DictWriter(f_global, fieldnames = fieldnames1)
        writer.writeheader()
        writer1.writeheader()
        t0 = 0
        t1 = 0
        for i in range(1,nfold+1):
            t0 = time.time()

            class Traindataset(Dataset):
                def __init__(self):
                    self.data=trainfolds[i-1]
                    self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1])).to(device)
                    self.len=self.data.shape[0]
                    self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]])).to(device)
                def __getitem__(self, index):
                    return self.x_data[index], self.y_data[index]
                def __len__(self):
                    return self.len
            class Testdataset(Dataset):
                def __init__(self):
                    self.data=testfolds[i-1]
                    self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1])).to(device)
                    self.len=self.data.shape[0]
                    self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]])).to(device)
                def __getitem__(self, index):
                    return self.x_data[index], self.y_data[index]
                def __len__(self):
                    return self.len

            traindataset = Traindataset()
            testdataset = Testdataset()

            header(model_lst,k,i,traindataset,testdataset)

            train_sampler,dev_sampler,test_sampler=dev_shuffle(shuffle_train,shuffle_test,val_split,traindataset,testdataset)
            #loaders
            train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, 
                                                       sampler=train_sampler,drop_last=True)
            dev_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, 
                                                       sampler=dev_sampler,drop_last=True)
            test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                                            sampler=test_sampler,drop_last=True)

            if k==0:
                model=Model0().to(device)
            if k==1:
                model=Model1().to(device)
            if k==2:
                model=Model2().to(device)
            if k==3:
                model=Model3().to(device)
            if k==4:
                model=Model4().to(device)
            if k==5:
                model=Model5().to(device)
            if k==6:
                model=Model6().to(device)
            
            criterion = nn.BCELoss(size_average=True)
            optimizer = torch.optim.SGD(model.parameters(), lr)    
            msg = 'Accuracy on test set before training: '+str(accuracy(test_loader))+'\n'
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
                    inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                    y_pred = model(inputs.to(device))
                    loss = criterion(y_pred.to(device), labels.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    #print accuracy ever l mini-batches
                    if l % 1000 == 999:
                        msg = '[%d, %5d] loss: %.3f' %(epoch + 1, l + 1, running_loss / 999)
                        print(msg)
                        logfile.write(msg + "\n")
                        running_loss = 0.0
                        #msg = 'Accuracy on dev set:' + str(accuracy(dev_loader))
                        #print(msg)
                        #logfile.write(msg + "\n")        
                accdev = (accuracy(dev_loader))
                msg = 'Accuracy on dev set:' + str(accdev)
                print(msg)
                logfile.write(msg + "\n")        
                is_best = bool(accdev > best_acc_dev)
                best_acc_dev = (max(accdev, best_acc_dev))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc_dev': best_acc_dev
                }, is_best,folder+'/F'+str(i)+'best.pth.tar')
                if is_best:
                    patience=0
                else:
                    patience = patience+1
                epoch = epoch+1
            state = torch.load(folder+'/F'+str(i)+'best.pth.tar')
            stop_epoch = state['epoch']
            model.load_state_dict(state['state_dict'])
            accuracy_dev = state['best_acc_dev']
            model.eval()
            acctest = (accuracy(test_loader))
            accs[i-1] = acctest
            precision,recall,f1_score = pre_rec(test_loader)
            precisions[i-1] = precision
            recalls[i-1] = recall
            f1_scores[i-1] = f1_score
            accs_dev[i-1] = accuracy_dev
            writer.writerow({'Fold': i, 'Accuracy': acctest,'Precision': precision,'Recall': recall,'F1_score': f1_score,'Stop_epoch': stop_epoch,'Accuracy_dev': accuracy_dev})
            table.column_headers = fieldnames
            table.append_row([i,acctest,precision,recall,f1_score,stop_epoch,accuracy_dev])
            print(table)
            print('----------------------------------------------------------------------')
            t1 = time.time()
            times[i-1] = int(t1-t0)
        duration = str(datetime.timedelta(seconds=np.sum(times)))
        writer.writerow({})
        writer.writerow({'Fold': 'Elapsed time: '+duration})
        avg_a,avg_p,avg_r,avg_f,avg_a_d=averages(accs,precisions,recalls,f1_scores,accs_dev)
        std_a,std_p,std_r,std_f,std_a_d=stds(accs,precisions,recalls,f1_scores,accs_dev)
        writer1.writerow({model_lst[k]: 'Accuracy','Avg': avg_a,'Std_dev': std_a})
        writer1.writerow({model_lst[k]: 'Precision','Avg': avg_p,'Std_dev': std_p})
        writer1.writerow({model_lst[k]: 'Recall','Avg': avg_r,'Std_dev': std_r})
        writer1.writerow({model_lst[k]: 'F1_score','Avg': avg_f,'Std_dev': std_f})
        writer1.writerow({model_lst[k]: 'Accuracy_dev','Avg': avg_a_d,'Std_dev': std_a_d})
        writer1.writerow({})
        writer1.writerow({model_lst[k]: 'Elapsed time: '+duration})
        avgtable.column_headers = fieldnames1
        avgtable.append_row(['Accuracy',avg_a,std_a])
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
        


# In[ ]:



