
# 按照最终提交的需求重跑Raw的CNN模型

import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn 
from torch.utils.data import Dataset, DataLoader, TensorDataset 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold

from torchvision import datasets 
from torchvision.transforms import ToTensor, Lambda

import numpy as np
from scipy import stats as ss 
from matplotlib import pyplot as plt 

import pickle

out = pd.read_csv("./processed_data/seqbrowse.csv")

amino_acids = "ACDEFGHIKLMNPQRSTVWXY-"


device = "cuda" if torch.cuda.is_available() else "cpu" # 根据环境配置设定device值为GPU还是CPU
# device = "cpu" 

print(f"Using {device} device")

#########################

# 按照10 20 30 40分别为155序列时的P20五分位点 将maxage分段用于cv拆分
out['maxagecls'] = pd.cut(out.maxage, bins=[-np.inf, 10,20,30,40,68, np.inf], right= False)
out['maxagecls'] = out.maxagecls.astype('str')
out.sort_values('maxage', ascending=False, inplace=True)

X = out['seq355'].values
# X = out['seq355'].apply(lambda x : x[25:275]).values
y = out['maxage'].values
y_cls = out['maxagecls'].values

y = y.reshape(-1, 1).astype(np.float32) 

seq_len = len(X[0]) 
def seq_encode(sequence, padding = seq_len):
    encoding = torch.zeros(len(amino_acids), padding).numpy()
    for i, aa in enumerate(sequence):
        encoding[amino_acids.index(aa), i] = 1
    return encoding

X_encode = np.array([seq_encode(seq) for seq in X]).astype(np.float32) 
# x_encode = np.expand_dims(x_encode, axis=1) # 插入一个新维度，以满足卷积时的分析要求



# 事先定义三个重要的超参数
learning_rate = 1e-3
batch_size = 16
epochs = 150

class LifespanPredictor(nn.Module):
    def __init__(self):
        super(LifespanPredictor, self).__init__() 
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            # nn.Conv2d(1, 32, 3),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 10 201
            # nn.Flatten(),
            nn.Linear( 9139 , 1024), # 22 * seq_len
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ) 
 
    def forward(self, x): 
        # x = x.view(x.size(0), -1) 
        #x = self.flatten(x) 
        x = self.linear_relu_stack(x) 
        return x


class EarlyStopping:
    def __init__(self, tolerance=10, min_delta=0): # 类参数init
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.validation_loss = 200000

    def __call__(self, train_loss, validation_loss):
        if self.validation_loss > validation_loss: # 验证集loss明显大于训练集
            self.validation_loss = validation_loss
            self.counter = 0 
        else:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

dfmodelcv = pd.DataFrame() # 初始化模型结果存储df
dfpred = pd.DataFrame() # 预测值存储df

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 同spacies分在一个fold, 不能用random_state=42参数了

cnt = 0
# Each fold is a tuple of (train_index, test_index)
for train_idx, test_idx in kf.split(X_encode, y_cls): # 用StratifiedGroupKFold则改为y_cls
    train_dataset = TensorDataset(torch.tensor(X_encode[train_idx], device = device), 
                                  torch.tensor(y[train_idx], device = device))
    test_dataset = TensorDataset(torch.tensor(X_encode[test_idx], device = device), 
                                 torch.tensor(y[test_idx], device = device))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # 实例化模型
    model = LifespanPredictor().to(device)

    # 定义损失函数和优化方法
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    early_stopping = EarlyStopping()

    epoch_accuracy = pd.DataFrame(columns = ['epoch', 'train_loss', 'validation_loss'])


    for epoch in range(epochs):
        running_loss = 0.0 # 初始化临时损失函数值
        for batch in train_dataloader: # 按照dataloader的批次分别取批次序号、X和y
            # Compute prediction and loss
            X_batch, y_batch = batch
            optimizer.zero_grad() # 将梯度清零
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            # validation_loss

            loss.backward() # 进行反向传播（以计算各节点的误差）
            optimizer.step() # 根据梯度调整参数
            
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader) # 计算训练集loss
        print(f"Epoch {epoch} loss: {train_loss}")

        outputs_val = model(torch.tensor(X_encode[test_idx], device = device)) # 计算验证集预测值
        validation_loss = loss_fn(outputs_val, torch.tensor(y[test_idx], device = device)).tolist() # 这里省略了.detach().numpy()
        
        epoch_accuracy = pd.concat([epoch_accuracy,
            pd.DataFrame({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss}, index = [0])],
            ignore_index=True
        ) # 改用concat以避开提示信息

        early_stopping(train_loss, validation_loss) # early_stopping是一个类对象，所以元素值会一直保留
        if early_stopping.early_stop:
            print("Early stopping\n Epoch: ", epoch)
            # break

    # 计算模型预测效果
    plt.close()
    with torch.no_grad():
        dfres = pd.DataFrame(
            {'pred' : model.cpu()(torch.tensor(X_encode[train_idx])).squeeze(-1),                                  
             'y' : y[train_idx].squeeze(-1),
            }) 
        plt.scatter(dfres.y, dfres.pred)
        train_corr, train_p = ss.spearmanr(dfres.y, dfres.pred)
        dfres = pd.DataFrame(
            {'pred' : model.cpu()(torch.tensor(X_encode[test_idx])).squeeze(-1),                                  
             'y' : y[test_idx].squeeze(-1),
            }) 
        test_corr, test_p = ss.spearmanr(dfres.y, dfres.pred) 
        dfpred = pd.concat([dfpred, dfres], ignore_index=True)
        plt.scatter(dfres.y, dfres.pred)
        test_pcorr, test_pp = ss.pearsonr(dfres.y, dfres.pred) 

    plt.title("model " + str(len(dfmodelcv)) + " : y vs pred")
    # plt.show() 启用则每次都需要手工关闭图片才会继续运行
    plt.savefig("Fig_corr_" + str(len(dfmodelcv)) + ".png"); plt.close()

    dfmodelcv = pd.concat([dfmodelcv,
                           pd.DataFrame({'stop_epoch' : epoch,
                                      'train_loss' : train_loss, 'test_loss' : validation_loss,
                                      'train_corr' : train_corr, 'train_p' : train_p, 
                                      'test_corr' : test_corr, 'test_p' : test_p,
                                      'test_pcorr' : test_pcorr, 'test_pp' : test_pp}, 
                                      index = [0])],
                           ignore_index=True
        ) # 存储模型考察信息

    with open(f'CNN_model_{cnt}.pkl', 'wb') as f:
        pickle.dump(model, f)

    cnt += 1


# 预测值绘图
plt.title(" y vs pred for testset")
plt.scatter(dfpred.y, dfpred.pred)
plt.savefig("Fig_pred.png"); plt.close()

# 模型信息汇总输出
print(dfmodelcv)
dfmodelcv.mean()
dfmodelcv.to_csv("dfmodel.csv")


# 绘制loss随epoch变化的线图
plt.plot(epoch_accuracy.train_loss)
plt.plot(epoch_accuracy.validation_loss)
plt.title("train_loss & validation_loss")
plt.show()

