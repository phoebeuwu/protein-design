


import pandas as pd
import numpy as np

############### 位点列表

# 569 102+39位点=103个 少了150，是104个 102+150+39
sitelst0 = [150, 52,48,51,63,54,55,53,46,35,57,47,60,30,28,27,50,43,38,62,67,64,49,75,81,255,59,32,214,204,227,141,37,78,218,144,145,42,167,36,101,166,170,210,195,259,151,193,40,65,258,89,197,179,160,238,155,206,216,231,156,161,164,234,184,242,271,99,207,236,119,211,33,250,212,61,272,79,143,269,91,237,233,219,232,31,208,249,205,239,168,246,224,83,222,228,153,253,162,185,29,159,261,39]

# 569 85个
sitelst0 = [52,48,51,63,54,55,53,46,35,57,47,60,30,28,27,50,43,38,62,67,64,49,75,81,255,59,32,214,204,227,141,37,78,218,144,145,42,167,36,101,166,170,210,195,259,151,193,40,65,258,89,197,179,160,238,155,206,216,231,156,161,164,234,184,242,271,99,207,236,119,211,33,250,212,61,272,79,143,269,91,237,233,219,232,31]

# 569 62个
sitelst0 = [52,48,51,63,54,55,53,46,35,57,47,60,30,28,27,50,43,38,62,67,64,49,75,81,255,59,32,214,204,227,141,37,78,218,144,145,42,167,36,101,166,170,210,195,259,151,193,40,65,258,89,197,179,160,238,155,206,216,231,156,161,164]

# 569 37个
sitelst0 = [52,48,51,63,54,55,53,46,35,57,47,60,30,28,27,50,43,38,62,67,64,49,75,81,255,59,32,214,204,227,141,37,78,218,144,145,42]


# 合并 50个
sitelst0 = [27,28,30,32,35,36,37,38,39,42,43,46,47,48,49,50,51,52,53,54,55,57,59,60,62,63,64,67,75,78,81,99,101,141,144,145,151,166,167,170,193,195,204,210,214,216,218,227,255,259]

# 合并 76个
sitelst0 = [27,28,30,32,33,35,36,37,38,39,40,42,43,46,47,48,49,50,51,52,53,54,55,57,59,60,62,63,64,65,67,75,78,81,83,89,99,101,119,141,144,145,150,151,155,156,160,161,164,166,167,168,170,179,193,195,197,204,206,207,208,210,211,212,214,216,218,219,227,231,234,237,238,255,258,259]





# 输出proteinmpnn_in_jax程序用的位置号

# 5X16 需要-1
list(np.array(sitelst0) - 1)
list(np.array(list(set(sitelst0))) - 1)

list(np.array(list(set(sitelst0))) )
############### 将半成品序列替换为完整序列


# 注意需要处理长度为355再保存 Human
out = pd.read_csv("./processed_data/seqbrowse.csv")
human_seq = out.loc[out.seqname == 'Homo_sapiens__NP_057623.2', 'seq355'].values[0]

def M_fillseq(dfin):
    print(len(dfin.seq[10]))
    headlen = 6 # 设定前面缺少的AA数量，5X16缺少前面5个aa
    tailstart = 299 # 设定后面缺少的AA起点，只有补全头部后只有299个
    dfin['seq'] = human_seq[:headlen] + dfin['seq'] + human_seq[tailstart:]
    print(len(dfin.seq[10]))
    return dfin




############# 准备好预测环境

from huggingface_hub import notebook_login
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch import nn
from transformers import AutoTokenizer
from evaluate import load
from datasets import Dataset
from sklearn import preprocessing

# 计算标化公式
out = pd.read_csv("./processed_data/seqbrowse.csv")
stdy = preprocessing.StandardScaler()
stdy.fit(out.loc[:, ['maxage']])

model_checkpoint = "facebook/esm2_t33_650M_UR50D"

def ESMpredict(test_seq, checkpointfile, batch_size = 16, model_checkpoint = model_checkpoint):
    # 根据指定的文件计算ESM预测值
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




# checkpointlst = ['t330', 't331', 't332', 't333', 't334']
checkpointlst = ['t330n', 't331n', 't332n', 't333n', 't334n']
resfilename = "temp.zip"
for name in checkpointlst:
    dftest[f'pred{name[-2:-1]}'] = ESMpredict(dftest.seq.tolist(), name)
    dftest.to_pickle(resfilename)

dftest['pred'] = (dftest['pred0'] + dftest['pred1'] + dftest['pred2']+ dftest['pred3']+ dftest['pred4'])/5

dftest.to_pickle(resfilename)



ESMpredict(lst, 't330')
ESMpredict(lst, 't331')
ESMpredict(lst, 't332')
ESMpredict(lst, 't333')
ESMpredict(lst, 't334')







############## 全部原始数据整合


# 生成124的4个类别的序列
dftest1 = pd.read_pickle("genmpnn/mpnn_30_1010.zip"); dftest1['cls'] = '30'
dftest2 = pd.read_pickle("genmpnn/mpnn_35_1010.zip"); dftest2['cls'] = '35'
dftest3 = pd.read_pickle("genmpnn/mpnn_51_1010.zip"); dftest3['cls'] = '51'
dftest4 = pd.read_pickle("genmpnn/mpnn_77_1010.zip"); dftest4['cls'] = '77'

dftest = pd.concat([dftest1, dftest2, dftest3, dftest4], ignore_index=True) 
dftest = M_fillseq(dftest)
dftest.to_pickle("dftest4cls_1010.zip")









####### Mouse

dfres = pd.read_pickle("processed_data/dfmouseres.zip")
df2 = pd.read_pickle("processed_data/dfresmse_h_m.zip"); 
df3 = pd.read_pickle("processed_data/dfresentropy_h_m.zip"); 

dfres = pd.concat([dfres, df2, df3], ignore_index=True)

dfres = dfres.groupby("seq").first().reset_index(drop = False)

dfres.cls.value_counts()

dfres = dfres.loc[dfres.sampling_temp <=1,:]

dfhigh = dfres.loc[(dfres.pred >= 10) & (dfres.cls == "ENTROPY"),:].sort_values('pred', ascending=False).reset_index(drop = True)
dfres = dfhigh




####### Beaver
dfres = pd.read_pickle("processed_data/dfres6grp_beaver.zip")
df2 = pd.read_pickle("processed_data/dfall2res_b.zip"); df2['cls'] = 'ALL'
df3 = pd.read_pickle("processed_data/dftestallres_b.zip"); df3['cls'] = 'ALL'
df4 = pd.read_pickle("processed_data/dfmse3res_b.zip"); df4['cls'] = 'MSE'
df5 = pd.read_pickle("processed_data/dfmse4res_b.zip"); df5['cls'] = 'MSE'
df6 = pd.read_pickle("processed_data/dfresmse_h_b.zip"); df6['cls'] = 'MSE'
df7 = pd.read_pickle("processed_data/dfresmse_h1_b.zip"); df6['cls'] = 'MSE'


dfres = pd.concat([dfres, df2, df3, df4, df5, df6, df7], ignore_index=True)

dfres = dfres.groupby("seq").first().reset_index(drop = False)

df6.sampling_temp.value_counts()

# dfres.to_pickle("processed_data/dftest6grppred.zip")

refline = 48.96 
plt.rcParams["font.family"] = "Microsoft YaHei"
sns.boxplot(y ="pred", x = "cls", data = dfres.loc[dfres.cls != 'human',:])
plt.axhline(y=refline, c='r', ls='dashed', lw=2)
plt.show()

dfhigh = dfres.loc[dfres.pred >= refline + 4,:]
dfhigh.cls.value_counts()

dfhigh = dfres.loc[(dfres.pred >= 29) & (dfres.cls == "MSE"),:].sort_values('pred', ascending=False).reset_index(drop = True)
dfres = dfhigh

dfhigh2 = dfres.loc[(dfres.pred >= 28) & (dfres.cls == "ENTROPY"),:].sort_values('pred', ascending=False).reset_index(drop = True)

dfres = pd.concat([dfhigh, dfhigh2], ignore_index=True)

dfhigh = dfhigh.loc[dfhigh.sampling_temp <1.5,:].reset_index(drop = True)

dfhigh.sampling_temp.value_counts()
dfhigh.to_csv("temp.csv")


dfhigh = dfres.loc[(dfres.pred >= 29) & (dfres.sampling_temp <=1.5),:]
dfhigh = dfhigh.loc[dfhigh.cls != 'Control', :].sort_values('pred', ascending=False).reset_index(drop = True)


dfres = dfhigh





########## 用递进方式减少计算量

dftest = dfhigh
resfilename = "dftestallres.zip"


# reflimit = [102.904594, 60.345097, 58.087387, 79.515976, 148.323853]
# reflimit = [40.65, 21.66, 21.71, 22.30, 23.53]
upcnt = 0.5

dftest['pred0'] = ESMpredict(dftest.seq.tolist(), 't330')
dftest = dftest.loc[dftest.pred0 > reflimit[0] + upcnt].copy(); dftest.to_pickle(resfilename)

dftest['pred1'] = ESMpredict(dftest.seq.tolist(), 't331')
dftest = dftest.loc[dftest.pred1 > reflimit[1] + upcnt].copy(); dftest.to_pickle(resfilename)

dftest['pred2'] = ESMpredict(dftest.seq.tolist(), 't332')
dftest = dftest.loc[dftest.pred2 > reflimit[2] + upcnt].copy(); dftest.to_pickle(resfilename)

dftest['pred3'] = ESMpredict(dftest.seq.tolist(), 't333')
dftest = dftest.loc[dftest.pred3 > reflimit[3] + upcnt].copy(); dftest.to_pickle(resfilename)

dftest['pred4'] = ESMpredict(dftest.seq.tolist(), 't334')
dftest = dftest.loc[dftest.pred4 > reflimit[4] + upcnt].copy()

dftest['pred'] = (dftest['pred0'] + dftest['pred1'] + dftest['pred2']+ dftest['pred3'] + dftest['pred4'])/5
dftest.to_pickle(resfilename)





# 这里生成的569
dfres1 = pd.read_pickle("processed_data/dfres0708_159.zip")
dfres2 = pd.read_pickle("processed_data/dfres0708_159a.zip"); dfres2['cls'] = '159'
dfres3 = pd.read_pickle("processed_data/dfres0708_159c.zip")

dfres4 = pd.read_pickle("processed_data/dfres0709_159.zip")
dfres5 = pd.read_pickle("processed_data/dfres0709_159a.zip")

dfres = pd.concat([dfres1, dfres2, dfres3, dfres4, dfres5], ignore_index=True) # 这个就是保存的dftopseq569.zip


