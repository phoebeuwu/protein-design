

'''
# 挂接google云盘
from google.colab import drive
drive.mount('/content/drive')
'''

import pandas as pd
import numpy as np
import math
import zipfile


rawcsv = "./processed_data/aligned138.csv"
resultdir = "/content/drive/MyDrive/result/"
homo_main_seq_name = "Homo_sapiens__NP_057623.2" # 人类主序列名称

dfseq = pd.read_csv(rawcsv)
MSA_len = len(dfseq['sequence'][0]) # MSA后全长


def seq_poscode(sequence, amino_acids0 = "ACDEFGHIKLMNPQRSTVWXY"): # 从0开始，按原始顺序标识MSA后序列中的原始aa位置
    # seq_poscode("--K-QR--")
    # seq0 = sequence.translate(str.maketrans('', '', '-')) # 转换回原始序列
    cnt = 0
    encoding = np.full(len(sequence), np.nan)
    for i, aa in enumerate(sequence):
        if aa in amino_acids0:
            encoding[i] = cnt
            cnt += 1
    return encoding

homo_seq = dfseq.loc[dfseq.seqname == homo_main_seq_name, 'sequence'].values[0] # 取人类主序列
homo_main_position = seq_poscode(homo_seq) # 计算人主序列aa位置备用


def mtxtrans(rawmtx, posarray, is3Dim = False, output3Dim = False, homo_main_position = homo_main_position): 
    # 按照给定的posarray将rawmtx数值填充至resmtx，各位点原先是几维，输出仍然是几维
    MSA_len = len(posarray)
    if is3Dim: # 是三维维矩阵，先展平
        dim2 = rawmtx.shape[1]; dim3 = rawmtx.shape[2]
        resdim = dim2 * dim3 
        rawmtx = np.reshape(rawmtx, (rawmtx.shape[0], resdim))
    else:
        resdim = rawmtx.shape[1]
    resmtx = np.full((MSA_len, resdim), np.nan) # 结果mtx
    for i in range(MSA_len):
        if not math.isnan(posarray[i]):
            resmtx[i] = rawmtx[int(posarray[i])] # 整行写入

    # 用homo_main_position位置删除不需要的数据块
    keeplst = []
    for i, pos in enumerate(homo_main_position):
        if not np.isnan(pos):
            keeplst.append(i) # 需要保留的位置
    resmtx = resmtx[np.ix_(keeplst)]

    # 用内插法填充所有的矩阵NA单元格
    resmtx = pd.DataFrame(resmtx).interpolate(limit_direction = "both").values

    if is3Dim & output3Dim:
        resmtx = np.reshape(resmtx, (resmtx.shape[0], dim2, dim3))

    resmtx = resmtx.astype(np.float16)

    return resmtx

def mtxtrans2(rawmtx, posarray, maincol = 0, homo_main_position = homo_main_position): 
    # 按照给定的posarray将形如(n, n, 288)的rawmtx数值填充至resmtx，各位点原先是几维，输出仍然是几维
    # maincol 指出固定长度的是第几列，默认第0列，即layer5矩阵的方式
    MSA_len = len(posarray)
    if maincol == 0: # 固定长度的是0列，换一下位置方便填充操作
        rawmtx = np.transpose(rawmtx, [1,2,0])
    resmtx = np.full((MSA_len, MSA_len, rawmtx.shape[2]), np.nan) # 内插太困难，考虑直接填充0.0算了
    for i in range(MSA_len):
        if not math.isnan(posarray[i]):
            for j in range(MSA_len):
                if not math.isnan(posarray[j]):
                    resmtx[i, j] = rawmtx[int(posarray[i]), int(posarray[j])] # 非对称阵

    # 用homo_main_position位置删除不需要的数据块
    keeplst = []
    for i, pos in enumerate(homo_main_position):
        if not np.isnan(pos):
            keeplst.append(i) # 需要保留的位置
    
    resmtx = resmtx[np.ix_(keeplst, keeplst)]

    # 内插后矩阵太大，layer4_1矩阵 600M/条，暂时先不做缺失值填充，后期统一处理

    return resmtx.astype(np.float16) # 更改为float16以降低内存用量，对保存文件大小无影响



####### 主程序段
layer4dict = {}; layer4dict1 = {} # 结果dict
layer5dict0 = {}; layer5dict1 = {}; layer5dict2 = {}; layer5dict3 = {} # 结果dict
layer6dict = {}
for i in range(len(dfseq)) : # 用循环依次处理每一个序列
    seqname = dfseq['seqname'][i] # seq名称
    seq = dfseq['sequence'][i] # MSA后序列
    seq_len = dfseq.seq_len[i] # 原始序列长度

    posarray = seq_poscode(seq)

    try:
        # 取npz矩阵文件
        zfile = zipfile.ZipFile(f'{resultdir}{seqname}.zip', 'r')
        NUll = zfile.extract(f'{seqname}/layer_4.npz', path=resultdir) # zfile.namelist()
        NUll = zfile.extract(f'{seqname}/layer_5.npz', path=resultdir) # zfile.namelist()
        NUll = zfile.extract(f'{seqname}/layer_6.npz', path=resultdir) # zfile.namelist()
        zfile.close()

        npz = np.load(f'{resultdir}{seqname}/layer_4.npz', allow_pickle= True) # 拆解layer4矩阵

        rawmtx0 = npz['arr_0'][0].squeeze(0).numpy() # shape: (n, 384)
        rawmtx1 = npz['arr_0'][1].squeeze(0).numpy() # shape: (n, n, 288)
        rawmtx2 = npz['arr_0'][2].squeeze(0).numpy() # shape: (n, 3, 3)
        rawmtx3 = npz['arr_0'][3].squeeze(0).numpy() # shape: (n,)

        resmtx0 = mtxtrans(rawmtx0, posarray)
        resmtx1 = mtxtrans2(rawmtx1, posarray, maincol = 2) # 固定长度的是第2列
        resmtx2 = mtxtrans(rawmtx2, posarray, is3Dim = True)
        resmtx3 = mtxtrans(np.reshape(rawmtx3,(rawmtx3.shape[0],1)), posarray)

        # resmtx1 = np.reshape(resmtx1,(resmtx1.shape[0], resmtx1.shape[1] * resmtx1.shape[2]))
        resmtx = np.concatenate((resmtx3, resmtx2, resmtx0), axis= 1) # 矩阵合并, shape: (355, 394)
        
        layer4dict[seqname] = resmtx; layer4dict1[seqname] = resmtx1


        npz = np.load(f'{resultdir}{seqname}/layer_5.npz', allow_pickle= True) # 拆解layer5矩阵

        rawmtx0 = npz['arr_0'][0].squeeze(0).numpy() # shape: (43, n, n)
        rawmtx1 = npz['arr_0'][1].squeeze(0).numpy() # shape: (43, n, n)
        rawmtx2 = npz['arr_0'][2].squeeze(0).numpy() # shape: (43, n, n)
        rawmtx3 = npz['arr_0'][3].squeeze(0).numpy() # shape: (19, n, n)

        resmtx0 = mtxtrans2(rawmtx0, posarray)
        resmtx1 = mtxtrans2(rawmtx1, posarray)
        resmtx2 = mtxtrans2(rawmtx2, posarray)
        resmtx3 = mtxtrans2(rawmtx2, posarray)

        # resmtx1 = np.reshape(resmtx1,(resmtx1.shape[0], resmtx1.shape[1] * resmtx1.shape[2]))
        resmtx = np.concatenate((resmtx3, resmtx2, resmtx0), axis= 1) # 矩阵合并
        
        layer5dict0[seqname] = resmtx0; layer5dict1[seqname] = resmtx1
        layer5dict2[seqname] = resmtx2; layer5dict3[seqname] = resmtx3


        npz = np.load(f'{resultdir}{seqname}/layer_6.npz', allow_pickle= True) # 拆解layer6矩阵

        rawmtx0 = npz['arr_0'][0].squeeze(0).numpy() # shape: (n, 3, 3)
        rawmtx1 = npz['arr_0'][1].squeeze(0).numpy() # shape: (n,)

        resmtx0 = mtxtrans(rawmtx0, posarray, is3Dim = True)
        resmtx1 = mtxtrans(np.reshape(rawmtx1,(rawmtx1.shape[0],1)), posarray)

        resmtx = np.concatenate((resmtx1, resmtx0), axis= 1) # 矩阵合并
        
        layer6dict[seqname] = resmtx

        print(f"{i} {seqname} done.")
    except:
        # print("Error.")
        pass

# 保存所有结果dict
np.savez_compressed("layer4dict.npz", layer4dict)
np.savez_compressed("layer4dict1.npz", layer4dict1)
np.savez_compressed("layer5dict0.npz", layer5dict0)
np.savez_compressed("layer5dict1.npz", layer5dict1)
np.savez_compressed("layer5dict2.npz", layer5dict2)
np.savez_compressed("layer5dict3.npz", layer5dict3)
np.savez_compressed("layer6dict.npz", layer6dict)


# !zip -q -r layerdict.zip *.npz




######################## 读入并整合所有矩阵结果
def mergemtx(npzfilelst, seqnamelst, dim1 = 355, dim2 = 394, dim3 = 1): # 整合已计算出的矩阵dict，合并为一个mtx    
    resdict = {}
    for npzfile in npzfilelst:
        resdict.update(np.load(f"processed_data/colablayer/{npzfile}", allow_pickle=True)['arr_0'].tolist()) # 添加读入的结果dict

    resmtx = np.full((len(seqnamelst), dim1, dim2 * dim3), np.nan, dtype=np.float16)
    for i in range(len(seqnamelst)) : # 用循环依次处理每一个序列
        resmtx[i] = resdict[seqnamelst[i]]  
        del resdict[seqnamelst[i]]  
        print(f"{seqnamelst[i]} done.")

    return resmtx

dfseq = pd.read_csv(rawcsv) # 这里要严格确认顺序正确！

resmtx4 = mergemtx(npzfilelst = ["layer4dict_1.npz", "layer4dict_2.npz", "layer4dict_3.npz", "layer4dict_3a.npz"],
                   seqnamelst = dfseq.seqname, dim2 = 394)

resmtx6 = mergemtx(npzfilelst = ["layer6dict_1.npz", "layer6dict_2.npz", "layer6dict_3.npz", "layer6dict_3a.npz"],
                   seqnamelst = dfseq.seqname, dim2 = 10)

resmtx46 = np.concatenate((resmtx4, resmtx6), axis= 2) # 合并分析
np.savez_compressed("processed_data/resmtx46_138.npz", resmtx46)

# resmtx46_155 = resmtx46[np.ix_(dfseq.index[dfseq.mainseq == 1])]
# np.savez_compressed("resmtx46_155.npz", resmtx46_155)
# np.savez_compressed("resmtx46_sample.npz", resmtx46_155[:10])


resmtx4_1 = mergemtx(npzfilelst = ["layer4dict1_1a.npz", "layer4dict1_1b.npz", "layer4dict1_1c.npz", \
                    "layer4dict1_2a.npz", "layer4dict1_2b.npz", "layer4dict1_2c.npz", \
                    "layer4dict1_3a.npz", "layer4dict1_3b.npz", "layer4dict1_3c.npz"],
                   seqnamelst = dfseq.seqname[dfseq.mainseq == 1 ].values.tolist(), dim2 = 355, dim3 = 288)

np.savez_compressed("resmtx4_1.npz", resmtx4_1) # 155条序列的数据共约10G


############################ 填充layer4_1矩阵中的缺失值

rawmtx = np.load("resmtx4_1.npz", allow_pickle=True)['arr_0']
resmtx = np.reshape(rawmtx, (155,355,355,288))

for i in range(resmtx.shape[0]): # 用内插法填充所有的矩阵NA单元格
    print(i)
    for j in range(resmtx.shape[3]):
        dfresmtx = pd.DataFrame(resmtx[i,:,:,j])
        dfresmtx = dfresmtx.interpolate(axis = 1, limit_direction = "both")
        dfresmtx = dfresmtx.interpolate(axis = 0, limit_direction = "both") 
        resmtx[i,:,:,j] = dfresmtx.values

rawmtx= {} # 释放内存
np.savez_compressed("resmtx4_1filled.npz", resmtx)

#############################

