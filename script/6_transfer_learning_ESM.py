#! pip install transformers evaluate datasets requests pandas sklearn

import pandas as pd
import numpy as np
import torch

from huggingface_hub import notebook_login

# notebook_login()
# Checkpoint name	Num layers	Num parameters
# esm2_t48_15B_UR50D	48	15B
# esm2_t36_3B_UR50D	36	3B
# esm2_t33_650M_UR50D	33	650M
# esm2_t30_150M_UR50D	30	150M
# esm2_t12_35M_UR50D	12	35M
# esm2_t6_8M_UR50D	6	  8M


model_checkpoint = "facebook/esm2_t33_650M_UR50D"

from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

out = pd.read_csv("processed_data/seqbrowse.csv")


# 按照10 20 30 40分别为155序列时的P20五分位点 将maxage分段用于cv拆分
out['maxagecls'] = pd.cut(out.maxage, bins=[-np.inf, 10,20,30,40,68, np.inf], right= False)
out['maxagecls'] = out.maxagecls.astype('str')

y_cls = out['maxagecls'].values


from sklearn import preprocessing

stdy = preprocessing.StandardScaler()
stdy.fit(out.loc[:, ['maxage']])
y = stdy.transform(out[['maxage']]) # 按照相同方式标化

all_seq = out.seq355 # .tolist()
all_label = ((y-2)/3.5) # .tolist() # out.maxage / 125 # raw = stdy.inverse_transform(y*3.5 + 2)



from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 同spacies分在一个fold, 不能用random_state=42参数了

cnt = 0
# Each fold is a tuple of (train_index, test_index)
for train_idx, test_idx in kf.split(all_seq, y_cls): # 用StratifiedGroupKFold则改为y_cls

    for i, item in enumerate(train_idx): # 删除人序列
        if item == 44:
            residx = np.delete(train_idx, i)
            train_idx = residx

    for i, item in enumerate(test_idx): # 删除人序列
        if item == 44:
            residx = np.delete(test_idx, i)
            test_idx = residx

    if cnt == 3: # 将67岁的133序列加入验证集
        for i, item in enumerate(train_idx): # 删除133序列
            if item == 133:
                residx = np.delete(train_idx, i)
                train_idx = residx
        test_idx = np.insert(test_idx, 25, 133)

    if cnt == 4: # 将67岁的133序列剔除出验证集
        for i, item in enumerate(test_idx): # 删除133序列
            if item == 133:
                residx = np.delete(test_idx, i)
                test_idx = residx
        train_idx = np.insert(train_idx, 106, 133)

    train_seq = all_seq[train_idx].tolist(); test_seq = all_seq[test_idx].tolist()
    train_label = all_label[train_idx].tolist(); test_label = all_label[test_idx].tolist()

    if cnt == 1 :
        break
    else:
        cnt += 1


# train_seq, test_seq, train_label, test_label = train_test_split(all_seq, all_label, test_size=0.2, random_state=42)

train_encodings = tokenizer(train_seq)
test_encodings = tokenizer(test_seq)

# train_encodings

## Create dataset
from datasets import Dataset

train_dataset = Dataset.from_dict(train_encodings)
test_dataset = Dataset.from_dict(test_encodings)

# add label
train_dataset = train_dataset.add_column("labels", train_label)
test_dataset = test_dataset.add_column("labels", test_label)

test_dataset
train_dataset
## load ESMfold model

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch import nn

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1, ignore_mismatched_sizes=True)
# model = AutoModelForSequenceClassification.from_pretrained("ESMmodel/checkpoint-t331n", num_labels=1, ignore_mismatched_sizes=True)

## Training
new_model_name = "finetuned_esm_fold"
batch_size = 1

args = TrainingArguments(
    output_dir = "esm_fineturne",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    save_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # num_train_epochs=100,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    no_cuda=False,
    greater_is_better = False,
)

from evaluate import load

metric = load("mse")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

import numpy as np

# 1 - 0.022982699796557426 / np.var(np.array(test_label)) # t333n 0.6283479551168543 334n 0.5040563681536321


trainer.evaluate(test_dataset)

pred = trainer.predict(test_dataset)
train_pred = trainer.predict(train_dataset)



from scipy import stats as ss 

ss.pearsonr(train_pred.label_ids.ravel(), train_pred.predictions.ravel()) 
ss.spearmanr(train_pred.label_ids, train_pred.predictions) 

ss.pearsonr(pred.label_ids.ravel(), pred.predictions.ravel()) 
ss.spearmanr(pred.label_ids, pred.predictions) 



dfpred1 = pd.DataFrame({'y' : stdy.inverse_transform(pred.label_ids * 3.5 + 2).ravel(), 
              'y_pred' : stdy.inverse_transform(pred.predictions * 3.5 + 2).ravel()
})
dfpred1.to_csv("t33pred4n.csv", index = False)

from matplotlib import pyplot as plt 
import seaborn as sns
sns.set() 

plt.scatter(stdy.inverse_transform(train_pred.label_ids * 3.5 + 2).ravel(), stdy.inverse_transform(train_pred.predictions * 3.5 + 2).ravel()) 
plt.scatter(stdy.inverse_transform(pred.label_ids * 3.5 + 2).ravel(), stdy.inverse_transform(pred.predictions * 3.5 + 2).ravel()) 
plt.show()


trainer.save_model("temp.mod") # 存储需要的模型
# 加载存储的模型
model = AutoModelForSequenceClassification.from_pretrained("esm_fineturne/checkpoint-770", num_labels=1, ignore_mismatched_sizes=True)

trainer2 = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer2.evaluate(test_dataset)
trainer2.predict(test_dataset)


len(train_seq[0])