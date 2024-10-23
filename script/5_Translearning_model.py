

############## 整合各种可用的数据到一个框架里面进行拟合

import pandas as pd
import numpy as np
import math
import zipfile


dfana = pd.read_csv("./processed_data/aligned138.csv")
dfaadummy = pd.read_csv("processed_data/aadummy.csv")

arrana = np.zeros((138,6,355,384)).astype(np.float32)

# 0 原始哑变量
arrana[:,0,:,:22] = dfaadummy[dfana.mainseq == 1].to_numpy()[:,1:].reshape((138,355,22)) # 哑变量矩阵

# 1 距离阵
dfdist = pd.read_pickle("processed_data/dfdist.zip")

dfsite = pd.DataFrame({"varname" : dfdist.columns[1:]})
dfsite['headfont'] = dfsite.varname.apply(lambda x : x[0])
dfsite['site1'] = dfsite.varname.apply(lambda x : x[1:x.find("_")])
dfsite['site2'] = dfsite.varname.apply(lambda x : x[x.find("_")+1:])

for i in range(len(dfsite)):
    arrana[:,1, int(dfsite.loc[i, 'site1']), int(dfsite.loc[i, 'site2'])] = dfdist.loc[:, dfsite.loc[i,'varname']]

# 2 角度两列
dfangle = pd.read_pickle("processed_data/dfangle.zip")
anglemtx = np.zeros((138, 355,2))
for i in range(355):
    anglemtx[:,i,0] = dfangle.loc[:,f"phi_{str(i)}"]
    anglemtx[:,i,1] = dfangle.loc[:,f"psi_{str(i)}"]
arrana[:,2,:,0] = anglemtx[:,:,0]
arrana[:,2,:,10] = anglemtx[:,:,1]

# 3 layer4 & 6
arr46 = np.load("./processed_data/resmtx46_138.npz", allow_pickle=True)['arr_0']

arrana[:,3,:,0] = arr46[:,:,0] # layer6_1
arrana[:,3,:,10:19] = arr46[:,:,1:10] # layer6_0 3*3
arrana[:,3,:,20] = arr46[:,:,10] # layer4_3
arrana[:,3,:,30:39] = arr46[:,:,11:20] # layer4_2 3*3

# 4 layer4_0
arrana[:,4,:,:] = arr46[:,:,20:] # layer4_0 384

np.savez_compressed("processed_data/transana.npz", arrana) # 下面有读入代码

################ 下面是分析模型

import torch
import torch.nn.functional as F
from torch import nn 
from torch.utils.data import Dataset, DataLoader, TensorDataset 
from torch.cuda.amp import autocast as autocast, GradScaler

from sklearn.model_selection import train_test_split 
from sklearn import preprocessing

from torchvision import datasets 
from torchvision.transforms import ToTensor, Lambda

import numpy as np
from scipy import stats as ss 
from matplotlib import pyplot as plt 


device = "cuda" if torch.cuda.is_available() else "cpu" # 根据环境配置设定device值为GPU还是CPU
# device = "cpu" 

print(f"Using {device} device")

######################### X矩阵准备

arrana = np.load("processed_data/transana.npz", allow_pickle=True)['arr_0']

X = arrana[:,0,:,:22] # 原始355哑变量

# 0 1 2 三层展平
X = arrana[:,0,:,:].copy()
X[:,:,22:22+355] = arrana[:,1,:,:355]
X[:,:,22+355] = arrana[:,2,:,0]
X[:,:,22+355+1] = arrana[:,2,:,10]

# 1 2 dist+angle展平
X = arrana[:,1,:,:].copy()
X[:,:,355+10] = arrana[:,2,:,0]
X[:,:,355+20] = arrana[:,2,:,10]


# 0 1 2 三层
X = np.zeros((138,6,355,100))
X[:,:3,:,:] = arrana[:,:3,:,:100] # 原始355哑变量 dist0 angle
X[:,3,:,:] = arrana[:,1,:,100:200] # dist1
X[:,4,:,:] = arrana[:,1,:,200:300] # dist2
X[:,5,:,:84] = arrana[:,1,:,300:384] # dist3

# 全部
X = np.zeros((138,9,355,200))
X[:,:6,:,:] = arrana[:,:6,:,:200] # 前200
X[:,7,:,:184] = arrana[:,1,:,200:384] # dist2
X[:,8,:,:155] = arrana[:,1,:,200:355] # layer4_0 2


# 只加上入选的变量
X = np.zeros((138,8000))
X[:,:7810]  = arrana[:,0,:,:22].reshape((138,-1))
X[:, 8000-74:8000 ] = pd.read_csv("processed_data/screeningvar138.csv").loc[:, pd.read_csv("processed_data/enet_var74.csv").varname]


##########################

dfana = pd.read_csv("processed_data/ana2260.csv")
X = dfana.iloc[:,3:].to_numpy()


y = dfana['maxage'].valuesy.reshape(-1, 1).astype(np.float16) 

# 按照10 20 30 40分别为155序列时的P20五分位点 将maxage分段用于cv拆分
dfana['maxagecls'] = pd.cut(dfana.maxage, bins=[-np.inf, 10,20,30,40,68, np.inf], right= False)
dfana['maxagecls'] = dfana.maxagecls.astype('str')
y_cls = dfana['maxagecls'].values

y = y.reshape(-1, 1).astype(np.float32) 

x_encode = X.astype(np.float32)
# x_encode = np.expand_dims(x_encode, axis=1)

x = X.astype(np.float32)
y = y.astype(np.float32)
maxseqlen = 355


# 事先定义三个重要的超参数
learning_rate = 1e-3
batch_size = 32
epochs = 50

class LifespanPredictor(nn.Module):
    def __init__(self):
        super(LifespanPredictor, self).__init__() 
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 176 191
            nn.Flatten(),
            nn.Linear(176 * 191 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        ) 
 
    #@autocast()
    def forward(self, x): 
        # x = x.view(x.size(0), -1) 
        # x = self.flatten(x) 
        x = self.linear_relu_stack(x) 
        return x
