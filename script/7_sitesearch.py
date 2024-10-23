
import pandas as pd
import numpy as np


# 全部为负的101个
sitelst0 = [48,52,54,63,47,55,46,51,49,60,62,50,64,53,67,57,35,28,30,27,43,255,78,81,39,195,38,218,227,37,59,75,101,145,141,216,151,210,144,170,42,214,193,32,99,36,258,40,155,259,167,168,231,156,166,197,207,33,119,212,206,237,238,234,89,150,179,65,160,219,208,204,211,83,79,232,269,271,184,91,249,236,233,272,185,222,164,143,242,61,250,246,41,224,153,253,162,161,247,239,268]

sitelst0 = list(np.array(list(set(sitelst0))) )

out = pd.read_csv("./processed_data/seqbrowse.csv")
human_seq = out.loc[out.seqname == 'Homo_sapiens__NP_057623.2', 'seq355'].values[0]

# dfana = pd.read_pickle('dftopseq397.zip') 
dfana = pd.read_pickle('processed_data/dftopseq569.zip') 
dfana['grp'] = 'RAW'

############# 准备好预测环境
from huggingface_hub import notebook_login
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch import nn
from transformers import AutoTokenizer
from evaluate import load
from datasets import Dataset
from sklearn import preprocessing
import torch

# 计算标化公式
out = pd.read_csv("./processed_data/seqbrowse.csv")
stdy = preprocessing.StandardScaler()
stdy.fit(out.loc[:, ['maxage']])

model_checkpoint = "facebook/esm2_t33_650M_UR50D"

def ESMpredict(test_seq, checkpointfile, batch_size = 32, model_checkpoint = model_checkpoint):
    # 根据指定的文件计算ESM预测值
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(f"ESMmodel/checkpoint-{checkpointfile}", num_labels=1, ignore_mismatched_sizes=True)
    metric = load("mse")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return metric.compute(predictions=predictions, references=labels)

    args = TrainingArguments(
        output_dir = "esm_fineturne",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        save_strategy="epoch",
        save_total_limit=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        no_cuda=False,
        greater_is_better = False,
    )

    trainer = Trainer(
        model,
        args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 设置预测文件
    test_encodings = tokenizer(test_seq); test_dataset = Dataset.from_dict(test_encodings)

    y_pred = trainer.predict(test_dataset); y_pred2 = stdy.inverse_transform(y_pred.predictions * 3.5 + 2); print(y_pred2.max())

    return y_pred2

def GetPred(dftest): # 返回需要的r预测值
    checkpointlst = ['t330n', 't331n', 't332n', 't333n', 't334n']
    # checkpointlst = ['t330n']
    for name in checkpointlst:
        dftest[f'pred{name[-2:-1]}'] = ESMpredict(dftest.seq.tolist(), name)
    dftest['pred'] = (dftest['pred0'] + dftest['pred1'] + dftest['pred2']+ dftest['pred3']+ dftest['pred4'])/5
    # dftest['pred'] = dftest['pred0'] 

    reflinelst = [41.174892,52.044838,62.01191,41.862404,46.317055]
    dftest['r0'] = (dftest.pred0 / reflinelst[0]) - 1
    dftest['r1'] = (dftest.pred1 / reflinelst[1]) - 1
    dftest['r2'] = (dftest.pred2 / reflinelst[2]) - 1
    dftest['r3'] = (dftest.pred3 / reflinelst[3]) - 1
    dftest['r4'] = (dftest.pred4 / reflinelst[4]) - 1
    dftest['r'] = (dftest['r0'] + dftest['r1'] + dftest['r2']+ dftest['r3'] + dftest['r4'])/5
    # dftest['r'] = 1

    return [dftest, dftest.r.mean(), dftest.r.max()]




dfres = dfana.copy()
## 生成替换序列
for site in sitelst0:
    df1 = dfana.copy()
    df1['seq'] = df1.seq.apply(lambda x : x[: site] + human_seq[site : site+1] + x[site+1 :])
    df1['grp'] = str(site)

    dfres = pd.concat([dfres, df1], ignore_index=True)

dfout, avg, max = GetPred(dfres)

dfsum = pd.DataFrame(dfout[['grp', 'r']].groupby('grp').agg(['mean', 'max']))
dfsum.to_csv("sitesum569.csv")
dfout.to_pickle("sitescreen569.zip")



####### 按降序进行累积替换
out = pd.read_csv("./processed_data/seqbrowse.csv")
human_seq = out.loc[out.seqname == 'Homo_sapiens__NP_057623.2', 'seq355'].values[0]

dfana = pd.read_pickle('processed_data/dftopseq569.zip')
dfana['grp'] = 'RAW'

dfsite = pd.read_csv("processed_data/sitesum569.csv")
dfsite = dfsite.loc[dfsite.grp != 'RAW', :] # 去掉raw
dfsite = dfsite.sort_values('mean', ascending=False).reset_index(drop=True)

# dfsite = pd.read_csv("dfsite70.csv")

dfres = dfana.copy()
df1 = dfana.copy()
## 生成替换序列
for site in dfsite.grp.tolist():
    s = int(site)
    df1['seq'] = df1.seq.apply(lambda x : x[: s] + human_seq[s : s+1] + x[s+1 :])
    df1['grp'] = str(site)

    dfres = pd.concat([dfres, df1], ignore_index=True)

# dfres.to_pickle("sitesortscreen502.zip")

# dfres = pd.read_pickle("sitesortscreen502.zip")

dfout, avg, max = GetPred(dfres)

dfsum = pd.DataFrame(dfout[['grp', 'r']].groupby('grp').agg(['mean', 'max']))
dfsum.to_csv("sitesortsum569.csv")
dfout.to_pickle("sitesortscreen569.zip")




############ 按照后退法筛选位点，要求可以取中间结果继续计算

out = pd.read_csv("./processed_data/seqbrowse.csv")
human_seq = out.loc[out.seqname == 'Homo_sapiens__NP_057623.2', 'seq355'].values[0]

# 此处读入中间结果，对应需要修改sitelst
dfana = pd.read_pickle('processed_data/dftopseq569.zip') ############ dfana = dfana.loc[:5,:]

################# 指定需要计算的位点，其余位点一律替换回human_seq
# SIRT6a 124个位点
# sitelst = [27,28,29,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,52,54,55,56,57,58,59,60,62,63,64,67,75,78,80,83,87,89,91,92,98,103,105,107,119,137,141,145,148,151,155,159,160,164,166,167,170,179,181,184,190,193,197,198,203,204,205,206,207,210,212,214,215,216,217,218,219,222,223,224,226,227,228,229,230,231,233,234,236,237,238,239,241,242,243,244,245,246,247,248,249,250,252,253,254,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271]
# 159 前62个
sitelst = [52,48,51,63,54,55,53,46,35,57,47,60,30,28,27,50,43,38,62,67,64,49,75,81,255,59,32,214,204,227,141,37,78,218,144,145,42,167,36,101,166,170,210,195,259,151,193,40,65,258,89,197,179,160,238,155,206,216,231,156,161,164]


# 其余位点一律替换回human_seq
for site in range(20,280):
    if site not in sitelst:
        dfana['seq'] = dfana.seq.apply(lambda x : x[: site] + human_seq[site : site+1] + x[site+1 :])


# 指定位点计算的最大深度
maxlen = 15


dfdropsite = pd.DataFrame(); dfsum0 = pd.DataFrame(); dfsave = pd.DataFrame()

dfres0 = dfana.copy() # dfres始终作为出发df

## 生成替换序列
while len(sitelst) > 0:
    dfres = pd.DataFrame()

    if maxlen >= len(sitelst) : # 全部纳入计算
        tmplst = sitelst
    else:
        tmplst = sitelst[len(sitelst) - maxlen : ]

    for site in tmplst:
        df1 = dfres0.copy()
        df1['seq'] = df1.seq.apply(lambda x : x[: site] + human_seq[site : site+1] + x[site+1 :])
        df1['grp'] = str(site)

        dfres = pd.concat([dfres, df1], ignore_index=True)

    dfout, avg, max = GetPred(dfres)

    dfsum = pd.DataFrame(dfout[['grp', 'r']].groupby('grp').agg(['mean']))
    dfsum.columns = ['rmean']
    dfsum = dfsum.reset_index(drop = False)

    rmeanmax = dfsum.rmean.max()
    dropsite = int(dfsum.loc[dfsum.rmean == rmeanmax, 'grp'].tolist()[0])

    dfdropsite = pd.concat([dfdropsite, pd.DataFrame({'site': dropsite, 'rmean' : rmeanmax}, index = [0])], ignore_index = True)

    dfsum['dropsite'] = dropsite

    dfsum0 = pd.concat([dfsum0, dfsum], ignore_index=True)

    dfres0 = dfout.loc[dfout.grp == str(dropsite),:].reset_index(drop = True)

    dfsave = pd.concat([dfsave, dfres0], ignore_index=True)
    
    sitelst.remove(dropsite)
    print(f'剔除：{dropsite}，r均值：{rmeanmax}，剩余长度：{len(sitelst)}：{sitelst}')

    dfdropsite.to_csv("dfdropsite.csv", index = False) # 位点删除列表
    dfsum0.to_csv("dfsitesum.csv", index = False) # 所有位点的累积计算记录
    dfres0.to_pickle("dfres0.zip") # 保存当前分析用数据集
    dfsave.to_pickle("dfsave.zip") # 保存累积分析用数据集

