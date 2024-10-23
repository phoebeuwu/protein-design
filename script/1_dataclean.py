
################## 本代码为数据准备############

import pandas as pd
import numpy as np

############### 整合两个不同来源的寿命数据表

dfanage = pd.read_table("./data/anage_data.txt")
dfmetainfo = pd.read_csv("./data/mammal_meta_info.csv")

dfanage = dfanage[dfanage['Class'] == "Mammalia"] # 筛选出哺乳动物数据
dfanage['species_short'] = dfanage.Genus + "_" + dfanage.Species #生成种系名称
dfmetainfo['species_short'] = dfmetainfo["Species Latin Name"].apply(lambda x : x.replace(" ", "_"))

dfmeta = pd.merge(dfmetainfo, dfanage[['Common name', 'species_short', 'Maximum longevity (yrs)']], how = 'outer', on = 'species_short')

# 整合两个来源的最大寿命
dfmeta['maxage'] = dfmeta['Maximum Lifespan (yrs)']
dfmeta.loc[dfmeta['Maximum Lifespan (yrs)'].isnull(), 'maxage'] = dfmeta['Maximum longevity (yrs)']

dfmeta = dfmeta[dfmeta['maxage'].notnull()]
   
dfmeta.loc[dfmeta['Common name'].isnull(), 'Common name'] = dfmeta['Species Common Name'] # 补充通用名
             
dfmeta.to_csv('processed_data/cleanedmeta.csv', index=False) # dfmeta = pd.read_csv('processed_data/cleanedmeta.csv')



####### 读入并整理序列数据

NCBIseqfile = 'processed_data/sirt6_sequence_1307.csv'

dfseq = pd.read_csv(NCBIseqfile)

### 非sirt6序列清理
dfseq['clean'] = 0
dfseq['tmp'] = dfseq.name.apply(lambda x : str(x).upper())
dfseq.loc[(dfseq.tmp.apply(lambda x : x.find("SIRT") < 0)) | (dfseq.tmp.apply(lambda x : x.find("6") < 0)), "clean"] = 1
dfseq.loc[dfseq.tmp.apply(lambda x : x.find("SIRTUIN-7") > 0), "clean"] = 1

# dfseq.clean.value_counts()
dfseq = dfseq[dfseq.clean == 0]
del dfseq['clean']
del dfseq['tmp']

dfseq['species_short'] = dfseq['species_short'].apply(lambda x : str(x).replace(" ", "_")) # dfseq.to_csv("temp.csv")


################按照dfseq筛选主序列

tmpdf = pd.read_csv("data/Sirt6_refseq_protein202.fasta", header=None, sep = "aaa", engine='python')
tmpdf.columns = ['rawstr']

for i in range(len(tmpdf)):
    if tmpdf['rawstr'][i][0] == ">" :
        seqname =  tmpdf['rawstr'][i][1:]
    else:
        tmpdf.loc[i, 'sequence'] = tmpdf.loc[i, 'rawstr'] 
    tmpdf.loc[i,'seqname'] = str(seqname)

del tmpdf['rawstr']


rawgrp = tmpdf.groupby('seqname', sort = False)
rfseq = rawgrp.agg(sum) # 只有字符串列的情况下，sum函数自动转为合并字符串
rfseq.reset_index(drop = False, inplace = True )

rfseq['id'] = rfseq.seqname.apply(lambda x : x[:x.find(" ")])
rfseq['mainseq'] = 1 # rfseq.to_csv("temp.csv")


### 两个数据源合并
dfseq.columns = ['id', 'name', 'species', 'sequencebk', 'species_short']
dfseq = pd.merge(dfseq, rfseq[['id', 'mainseq', 'sequence']], how = 'outer', on = 'id')

# rfseq.to_csv('temp2.csv')

dfseq.loc[dfseq.id == 'XP_002928505.1', 'mainseq'] = 0
dfseq.loc[dfseq.id == 'NP_001401780.1', 'mainseq'] = 1 
dfseq.loc[dfseq.id == 'NP_001401780.1', 'sequence'] = dfseq.sequencebk #  替换 XP_002928505.1 为 NP_001401780.1

dfseq.loc[dfseq.id == 'XP_004395388.1', 'species_short'] = 'Odobenus_rosmarus_divergens'

dfseq1 = dfseq.loc[dfseq.mainseq == 1, :].copy() # dfseq1.to_csv("temp2.csv")

#### 顺便用anage筛选哺乳动物数据
dfall = pd.merge(dfseq1.loc[:, ['id', 'name', 'species', 'species_short', 'mainseq', 'sequence']], 
                 dfmeta[['species_short', "maxage", 'Common name']], how = 'inner', on = 'species_short')
dfall.rename( columns = {'Common name': 'common_name'}, inplace = True) 

dfall.insert(0,'seqname', dfall['species_short'] + "__" + dfall['id']) # 给每个序列单独名称



# 跨物种去重：相同序列首选ensemble的，如果没有就选寿命最长的一条
dfall.sort_values(["sequence", "mainseq", "maxage", "species_short"], inplace=True, ascending=[ True, False, False, True])
dfall = dfall.groupby(['sequence']).first() 
dfall.reset_index(drop=False, inplace=True) # dfall.to_csv("temp.csv")


dfall.insert(0, 'seq_len', dfall.sequence.apply(lambda x : len(x))) # 计算序列长度

dfall.loc[dfall.name.apply(lambda x : "PREDICTED" in x), 'predict'] = 1

dfall.to_csv('./processed_data/sirt6_sequence_for_MSA_1307.csv', index=False)


# 生成对齐用文件，现在只处理mainseq
tmpdat = dfall.copy()
tmpdat['tmpstr'] = tmpdat['seqname'].apply(lambda x : ">" + x)
tmpdat.sort_values('seq_len', inplace=True, ascending=False)
tmpdat = pd.DataFrame(tmpdat[['tmpstr', 'sequence']].stack())
tmpdat.to_csv("./for_msa.fasta", header = False, index=False)



########################

# 最后使用的是MEGA7环境下的muscle算法生成的MSA结果。

################




######## 读入已对齐的序列
# 将数据转换为序列占单行
tmpdf = pd.read_csv("./processed_data/msa_maga7_muscle142.fas", header=None)
tmpdf.columns = ['rawstr']

for i in range(len(tmpdf)):
    if tmpdf['rawstr'][i][0] == ">" :
        seqname =  tmpdf['rawstr'][i][1:]
    else:
        tmpdf.loc[i, 'sequence'] = tmpdf.loc[i, 'rawstr'] 
    tmpdf.loc[i,'seqname'] = str(seqname)

del tmpdf['rawstr']

rawgrp = tmpdf.groupby('seqname', sort = False)
aliseq = rawgrp.agg(sum) # 只有字符串列的情况下，sum函数自动转为合并字符串

# aliseq.to_csv('./processed_data/aliseq_1ps_processed.csv')

# 合并数据并存储备用
out = pd.merge(aliseq, dfall.rename(columns = {'sequence': 'sequence0'}),
               how="inner", on="seqname")
del out['sequence0']

def M_getpos(seq, amino_acids = "ACDEFGHIKLMNPQRSTVWXY"):
    # 检索序列在拼接后数据中的起始位置
    # seq = "-2MS---"
    start0 = 5000; end0 = 0
    for aa in amino_acids:
        start = seq.find(aa)
        if (start < start0) & (start >= 0):
            start0 = start
        end = seq.rfind(aa)
        if (end > end0) & (end >= 0):
            end0 = end
    return (start0, end0)

out['postuple'] = out.sequence.apply(lambda x : M_getpos(x))
out['startpos'] = out['postuple'].apply(lambda x : x[0])
out['endpos'] = out['postuple'].apply(lambda x : x[1])
del out['postuple']

out.sort_values("seqname", inplace=True)


out.loc[out.mainseq == 1, :].to_csv("./processed_data/aligned142.csv", index = False)