

# biopython
import Bio.PDB
from Bio.PDB import calc_dihedral
import pandas as pd
import numpy as np
import math
import zipfile

from matplotlib import pyplot as plt 
from scipy import stats as ss 

rawcsv = "./processed_data/aligned142.csv" # rawcsv = "./processed_data/aligned422.csv"
resultdir = "./pdb142/"
homo_main_seq_name = "Homo_sapiens__NP_057623.2" # 人类主序列名称

amino_acids = "ACDEFGHIKLMNPQRSTVWXY-"

dfseq = pd.read_csv(rawcsv)

MSA_len = len(dfseq['sequence'][0]) # MSA后全长


############# 计算哪些位点需要纳入分析，标准：50%以上序列非缺失，或属于人序列的位点

def seq_is_aa(sequence, MSA_len = MSA_len): # 标记哪些位置是aa
    encoding = np.zeros(MSA_len)
    for i, aa in enumerate(sequence):
        if aa == "-":
            encoding[i] = 0
        else:
            encoding[i] = 1
    return encoding

dfaapos = pd.DataFrame( [seq_is_aa(seq) for seq in dfseq.sequence] ) 
aapos = dfaapos.sum() / dfaapos.shape[0]

homo_seq = dfseq.loc[dfseq.seqname == homo_main_seq_name, 'sequence'].values[0] # 取人类主序列
homopos = seq_is_aa(homo_seq)

dfkeep = pd.DataFrame({'aa' : aapos, 'homo' : homopos})
dfkeep['keep'] = 0
dfkeep.loc[(dfkeep.aa >= 0.5) | (dfkeep.homo == 1), 'keep'] = 1
dfkeep.to_csv("./processed_data/dfkeep.csv") # dfkeep = pd.read_csv("./processed_data/dfkeep.csv")
# 如果用138条序列分析，则会有4个非人类位点需要纳入，不折腾了。
# 最终确认需要保留的位点就是人序列的355个，因此简化后面的代码为保留homo_main_position即可

# 生成最终分析结果用的df
dfbrowse = dfseq.loc[:, ['seqname', 'common_name', 'maxage', 'sequence']]
dfbrowse['seq355'] = ''
cnt = 0
for i, homo in enumerate(dfkeep.homo):
    if homo == 1:
        dfbrowse[f'p{cnt}'] = dfbrowse.sequence.apply(lambda x : x[i])
        dfbrowse['seq355'] = dfbrowse[['sequence', 'seq355']].apply(lambda x : x[1] + x[0][i], axis = 1)
        cnt += 1

# dfbrowse = pd.read_csv("processed_data/seqbrowse.csv")
dfbrowse['seq275'] =  dfbrowse['seq355'].apply(lambda x : x[24:275]) # 序列编号是24-275

dfbrowse.to_csv("processed_data/seqbrowse.csv", index=False) # dfbrowse.to_csv("processed_data/seqbrowse422.csv", index=False)
########################



''' 调试用程序段
i = 127 # Pan_troglodytes__PNI25174.1 最短，index为最后一个445 人为119，测试用90 Acinonyx_jubatus__XP_05307607
seqname = dfseq['seqname'][i] # seq名称
seq = dfseq['sequence'][i] # MSA后序列
seq_len = dfseq.seq_len[i] # 原始序列长度
sequence = seq

# 取pdb文件
zfile = zipfile.ZipFile(f'./colabresult/{seqname}.zip', 'r')
zfile.extract(f'{seqname}/pred_init.pdb', path='./colabresult') # zfile.namelist()
zfile.close()

pdb_file = f'{resultdir}{seqname}/pred_init.pdb' # 根据相应设定给出seq对应的pdb文件路径
'''

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

homo_main_position = seq_poscode(homo_seq) # 计算人主序列aa位置备用


######## # aa位点哑变量生成
def seq_aa_dummy(sequence, amino_acids0 = "ACDEFGHIKLMNPQRSTVWXY-", MSA_len = MSA_len): # aa位点哑变量生成
    encoding = np.zeros((MSA_len, len(amino_acids0))) # 改了一下顺序，让同一位点的22个变量放在一起
    for i, aa in enumerate(sequence):
        encoding[i, amino_acids.index(aa)] = 1
    return np.ravel(encoding)

namelst = []; keeplst = [] # 计算需要保留的位点所对应的变量名称
for i, pos in enumerate(homo_main_position):
    if np.isnan(pos):
        for j in range(len(amino_acids)):
            namelst.append(f'z{i}_{j}')
    else:
        for j in range(len(amino_acids)):
            namelst.append(f'p{int(pos)}{amino_acids[j]}')
            keeplst.append(f'p{int(pos)}{amino_acids[j]}')
namelst.insert(0, 'seqname'); keeplst.insert(0, 'seqname')


dfaadummy = pd.DataFrame()
for i in range(len(dfseq)) : # 用循环依次处理每一个序列
    seqname = dfseq['seqname'][i] # seq名称
    seq = dfseq['sequence'][i] # MSA后序列

    dfdummy = pd.DataFrame(seq_aa_dummy(seq)).T
    dfdummy.insert(0, 'seqname', seqname)
    dfaadummy = pd.concat([dfaadummy, dfdummy], ignore_index=True)

dfaadummy.columns = namelst
dfaadummy = dfaadummy[keeplst] # 只保留需要的变量
dfaadummy.to_csv("processed_data/aadummy.csv", index=False) # dfaadummy = pd.read_csv("processed_data/aadummy.csv")

#########


def M_transloop(sequence, pdb_file, seqname, usefulpos = []):
    # 依据PDB文件解析出aa间距离和夹角
    # sequence MSA后序列；pdb_file PDB文件路径；usefulpos 指定需要保留的位置

    # M_transloop()需要完成的基本步骤：
    # 1. 解析pdb
    # 2. 计算距离和夹角
    # 3. 长形格式转换为矩阵
    # 4. 计算MSA后aa序号
    # 5a. 待议：是否只取某些需要的位置数据
    # 5b. 内插扩充为MSA后矩阵
    # 6. 矩阵转换回长形格式并返回

    parser = Bio.PDB.PDBParser(QUIET=True) # PDBParser，QUIET 屏蔽每次都出现的警告输出

    structure = parser.get_structure("protein", pdb_file) # Structure

    model = structure[0]

    # 只有一个链，aa编号是从1开始
    # Get all the ca atoms
    ca_atoms = [];  c_atoms = [];  n_atoms = []; plddt_atoms = []
    for chain in model:
        # print(chain)
        for residue in chain:
            ca_atoms.append(residue["CA"])
            c_atoms.append(residue["C"])
            n_atoms.append(residue["N"])
        for ca in ca_atoms:
            plddt_atoms.append(ca.get_bfactor())           

    def compute_distance(ca_atoms): # 计算aa间距离，返回df
        distances = []
        for i in range(len(ca_atoms)):
            for j in range(len(ca_atoms)):
                distance = ca_atoms[i] - ca_atoms[j]
                distances.append((i, j, distance))
        return pd.DataFrame(distances, columns=["residue1", "residue2", "distance"])

    def compute_plddt(plddt_atoms): # 计算plddt乘积，返回df
        plddts = []
        for i in range(len(plddt_atoms)):
            for j in range(len(plddt_atoms)):
                plddt = plddt_atoms[i] * plddt_atoms[j]
                plddts.append((i, j, plddt))
        return pd.DataFrame(plddts, columns=["residue1", "residue2", "plddt"])


    def compute_angle(ca_atoms): # 计算aa的二面角，返回df
        angles = []
        maxlen = len(ca_atoms)
        for i in range(maxlen): # 头尾无法计算角度
            N = n_atoms[i].get_vector(); CA = ca_atoms[i].get_vector(); C = c_atoms[i].get_vector()
            if i == 0:
                phi = np.NaN
            else:
                CP = c_atoms[i-1].get_vector()
                phi = calc_dihedral(CP, N, CA, C) * -180 / np.pi # 输出为角度？

            if i == maxlen - 1:
                psi = np.NaN
            else:
                NA = n_atoms[i+1].get_vector()
                psi = calc_dihedral(N, CA, C, NA) * -180 / np.pi

            angles.append((i, phi, psi))

        return pd.DataFrame(angles, columns=["residue", "phi", "psi"])

    dfdistances = compute_distance(ca_atoms)
    dfplddts = compute_plddt(plddt_atoms)
    dfangles = compute_angle(ca_atoms)


    # 转换为矩阵格式    pd.DataFrame(anglesmtx).to_csv("mtx.csv")
    distmtx = np.zeros((seq_len, seq_len))
    for i in range(dfdistances.shape[0]):
        distmtx[dfdistances.residue1[i], dfdistances.residue2[i]] = dfdistances.distance[i]

    plddtmtx = np.zeros((seq_len, seq_len))
    for i in range(dfplddts.shape[0]):
        plddtmtx[dfplddts.residue1[i], dfplddts.residue2[i]] = dfplddts.plddt[i]

    # 这里只是借用原代码，方便修改
    phimtx = np.zeros((seq_len, seq_len)); psimtx = np.zeros((seq_len, seq_len))
    for i in range(dfangles.shape[0]): # 只有主对角线有数据
        phimtx[dfangles.residue[i], dfangles.residue[i]] = dfangles.phi[i]
        psimtx[dfangles.residue[i], dfangles.residue[i]] = dfangles.psi[i]



    def mtxtrans(rawmtx, posarray):
        # 按照给定的posarray将rawmtx数值填充至resmtx，角度阵虽然短1，但代码应该不用改
        # rawmtx = np.array([0,22,33,22,55,22,33,22,11]).reshape(3,3) # 原始mtx
        # posarray = np.array([np.nan, 0, np.nan, 1, np.nan, 2, np.nan]) 
        # resmtx = np.full((7,7), np.nan) # 原始位置idx
        MSA_len = len(posarray)
        resmtx = np.full((MSA_len, MSA_len), np.nan) # 结果mtx
        for i in range(MSA_len):
            if not math.isnan(posarray[i]):
                for j in range(i, MSA_len):
                    if not math.isnan(posarray[j]):
                        # print(i, j)
                        resmtx[i, j] = rawmtx[int(posarray[i]), int(posarray[j])]
                        resmtx[j, i] = rawmtx[int(posarray[j]), int(posarray[i])] # 这样非对称阵也可处理了
            # else:
            #    resmtx[i, i] = 0 # 主对角线必须为0，这里暂时不能写0，会影响后面的内插
        return resmtx

    posarray = seq_poscode(sequence)
    distresmtx = mtxtrans(distmtx , posarray) # 扩充为MSA后矩阵 
    plddtresmtx = mtxtrans(plddtmtx , posarray) # 扩充为MSA后矩阵 
    phiresmtx = mtxtrans(phimtx , posarray) # 扩充为MSA后矩阵 
    psiresmtx = mtxtrans(psimtx , posarray) # 扩充为MSA后矩阵 


    # 在这里用homo_main_position位置处理不需要的数据块，这样可以改善后续的内插结果
    def mtx_shrink(mtx, homo_main_position = homo_main_position):
        # mtx = distresmtx
        dfmtx = pd.DataFrame(mtx)
        namelst = []; keeplst = []
        for i, pos in enumerate(homo_main_position):
            # print(i, pos)
            if np.isnan(pos):
                namelst.append(f'z{i}')
            else:
                namelst.append(f'p{int(pos)}')
                keeplst.append(f'p{int(pos)}')
        
        dfmtx.columns = namelst
        dfmtx = dfmtx[keeplst]
        dfmtx = dfmtx.T
        dfmtx.columns = namelst
        dfmtx = dfmtx[keeplst]
        dfmtx = dfmtx.T

        return dfmtx

    dfdistresmtx = mtx_shrink(distresmtx) # dfdistresmtx.to_csv("resmtx.csv")
    dfplddtresmtx = mtx_shrink(plddtresmtx) # dfdistresmtx.to_csv("resmtx.csv")
    dfphiresmtx = mtx_shrink(phiresmtx) # dfangleresmtx.to_csv("resmtx.csv")
    dfpsiresmtx = mtx_shrink(psiresmtx) # dfangleresmtx.to_csv("resmtx.csv")


    # 用内插法填充所有的矩阵NA单元格，直接内插替换一个方向是对的，但直接做另外一个方向就会出错，需要预处理
    # 现在两端一律不处理，留空填充为均值
    dfdistresmtx = dfdistresmtx.interpolate(axis = 1, limit_direction = "both", limit_area = 'inside')  
    for i in range(dfdistresmtx.shape[0]):
        if np.isnan(dfdistresmtx.iloc[i, i]):
            dfdistresmtx.iloc[i, i] = 0
    dfdistresmtx = dfdistresmtx.interpolate(axis = 0, limit_direction = "both", limit_area = 'inside') # dfdistresmtx.to_csv("shrinkmtx.csv")


    def mtx2df(dfmtx, half = True, headfont = 'd'): # 将df矩阵转换回长形格式，half 对称阵转一半即可
        # half 只记录上三角即可 headfont 用于标记是距离还是角度d/a
        res = []
        for i in range(dfmtx.shape[0]):
            if half:
                if headfont == 'a':
                    startj = i + 2 #  角度数据，主对角线和相邻位点角度均无意义，不记录
                elif headfont == 't':
                    startj = i # 需要记录主对角线数据
                else:
                    startj = i + 1 #  主对角线无意义，不记录
            else:
                startj = 0 # 非对称阵，数据全部保留
            for j in range(startj, dfmtx.shape[1]):
                rowname = dfmtx.columns[i]; colname = dfmtx.columns[j]
                newname = f"{headfont}{rowname[1:]}_{colname[1:]}"
                res.append((newname, dfmtx.loc[rowname, colname]))
        return pd.DataFrame(res, columns=["varname", "value"])

    # 矩阵转换回长形格式
    dfdist = mtx2df(dfdistresmtx)
    dfplddt = mtx2df(dfplddtresmtx, headfont="t") # dfplddt.to_csv('temp.csv', index = False)

    phimtx = []
    for i in range(dfphiresmtx.shape[0]):
        phimtx.append((f"phi_{i}", dfphiresmtx.iloc[i,i]))
    dfphi = pd.DataFrame(phimtx, columns=["varname", "value"])

    psimtx = []
    for i in range(dfphiresmtx.shape[0]):
        psimtx.append((f"psi_{i}", dfpsiresmtx.iloc[i,i]))
    dfpsi = pd.DataFrame(psimtx, columns=["varname", "value"])


    dfdist = dfdist.set_index("varname").T
    dfdist.insert(0, 'seqname', seqname) # dfdist.T.to_csv("dfdist.csv")

    dfplddt = dfplddt.set_index("varname").T
    dfplddt.insert(0, 'seqname', seqname) # dfdist.to_csv("dfdist.csv")

    dfangle = pd.concat([dfphi,dfpsi], ignore_index= True)
    dfangle = dfangle.loc[dfangle.value.notnull(),:] # 删除无数据的case
    dfangle = dfangle.set_index("varname").T
    dfangle.insert(0, 'seqname', seqname) # dfangle.to_csv("dfangle.csv")

    return [dfdist, dfplddt, dfangle]


####### 主程序段
dfoutdist = pd.DataFrame(); dfoutplddt = pd.DataFrame(); dfoutangle = pd.DataFrame() # 3个结果df

for i in range(len(dfseq)) : # 用循环依次处理每一个序列
    seqname = dfseq['seqname'][i] # seq名称
    seq = dfseq['sequence'][i] # MSA后序列
    seq_len = dfseq.seq_len[i] # 原始序列长度

    try:
        # 取pdb文件
        # zfile = zipfile.ZipFile(f'{resultdir}{seqname}.zip', 'r')
        # NUll = zfile.extract(f'{seqname}/pred_init.pdb', path=resultdir) # zfile.namelist()
        # zfile.close()

        pdb_file = f'{resultdir}{seqname}.pdb' # 根据相应设定给出seq对应的pdb文件路径

        reslst = M_transloop(seq, pdb_file, seqname)

        dfoutdist = pd.concat([dfoutdist, reslst[0]], ignore_index=True)
        dfoutplddt = pd.concat([dfoutplddt, reslst[1]], ignore_index=True)
        dfoutangle = pd.concat([dfoutangle, reslst[2]], ignore_index=True)

        print(f"{i} {seqname} done.")
    except:
        # print("Error.")
        pass

dfdist = dfoutdist.copy()
dfplddt = dfoutplddt.copy()
dfangle = dfoutangle.copy()



# 两端的dist缺失值在这里做填充
for column in list(dfdist.columns[dfdist.isnull().sum() > 0]):
    mean_val = dfdist[column].mean()
    dfdist[column].fillna(mean_val, inplace=True)

# angle的缺失值这里做填充，两端点有数值是正常情况
for column in list(dfangle.columns[dfangle.isnull().sum() > 0]):
    mean_val = dfangle[column].mean()
    dfangle[column].fillna(mean_val, inplace=True)


dfdist.to_pickle("processed_data/dfdist.zip") # dfdist = pd.read_pickle("processed_data/dfdist.zip")
dfplddt.to_pickle("processed_data/dfplddt.zip") # dfplddt = pd.read_pickle("processed_data/dfplddt.zip")
dfangle.to_pickle("processed_data/dfangle.zip") # dfangle = pd.read_pickle("processed_data/dfangle.zip")


################## 用相关系数做变量筛选
def corrscreening(dfana, plimit = 0.05, keepleftvar = 1): # 根据P界值筛选变量，keepleftvar 保留左侧几个变量
    corrres = []
    for i in range(2, dfana.shape[1]): # 0列为name，1列为maxage
        if i % 20000 == 0:
            print(i)
        s, p = ss.pearsonr(dfana.maxage, dfana.iloc[:, i])
        corrres.append((dfana.columns[i], s, abs(s), p))
   
    dfres = pd.DataFrame(corrres)
    dfres.columns = ['varname', 'corr', 'abscorr', 'p']

    keeplst = list(dfana.columns[ : keepleftvar].values)
    keeplst.extend(dfres.loc[dfres.p <= plimit, 'varname'].values)
    dfkeep = dfana[keeplst]

    print(f"入选变量数：{len(keeplst)}")
 
    return [dfres, dfkeep]


#### 只取mainseq的程序段
dfanadummy = pd.merge(dfseq.loc[dfseq.mainseq == 1, ["seqname", "maxage"]], dfaadummy, on="seqname", how="inner")
dfanadist = pd.merge(dfseq.loc[dfseq.mainseq == 1, ["seqname", "maxage"]], dfdist, on="seqname", how="inner")
dfanaangle = pd.merge(dfseq.loc[dfseq.mainseq == 1, ["seqname", "maxage"]], dfangle, on="seqname", how="inner")
#### 

''' 全部进行分析的程序段
dfanadummy = pd.merge(dfseq[["seqname", "maxage"]], dfaadummy, on="seqname", how="inner")
dfanadist = pd.merge(dfseq[["seqname", "maxage"]], dfdist, on="seqname", how="inner")
dfanaangle = pd.merge(dfseq[["seqname", "maxage"]], dfangle, on="seqname", how="inner")
'''

dfcorrdummy, dfkeepdummy = corrscreening(dfanadummy, plimit = 0.01, keepleftvar = 2) # dfcorrdist.to_csv("temp.csv")
dfcorrdist, dfkeepdist = corrscreening(dfanadist, plimit = 0.01)
dfcorrangle, dfkeepangle = corrscreening(dfanaangle, plimit = 0.01)

dfcorrdummy.sort_values("p").to_csv("processed_data/corrdummy.csv", index=False)
dfcorrangle.sort_values("p").to_csv("processed_data/corrangle.csv", index=False)
dfcorrdist.sort_values("p").to_csv("processed_data/corrdist.csv", index=False) # dfcorrdist = pd.read_csv("processed_data/corrdist.csv")


dfkeep = pd.merge(dfkeepdummy, dfkeepdist, on='seqname')
dfkeep = pd.merge(dfkeep, dfkeepangle, on='seqname')
dfkeep.to_csv('processed_data/screeningvar142_11133.csv', index=False) # dfkeep = pd.read_csv("processed_data/screeningvar138.csv")


keepnum = 5000 # dist入选变量太多，只取前5000个
lst = dfcorrdist.sort_values("p").varname[:keepnum].values
lst.sort()
lst = np.insert(lst, 0, 'seqname')
dfkeepdist = dfkeepdist.loc[:,lst] 

dfkeep = pd.merge(dfkeepdummy, dfkeepdist, on='seqname')
dfkeep = pd.merge(dfkeep, dfkeepangle, on='seqname')
dfkeep.to_csv('processed_data/screeningvar142.csv', index=False) # dfkeep = pd.read_csv("processed_data/screeningvar138.csv")

# 保留2000个dist
keepnum = 2000
lst = dfcorrdist.sort_values("p").varname[:keepnum].values
lst.sort()
lst = np.insert(lst, 0, 'seqname')
dfkeepdist = dfkeepdist.loc[:,lst] 

dfkeep = pd.merge(dfkeepdummy, dfkeepdist, on='seqname')
dfkeep = pd.merge(dfkeep, dfkeepangle, on='seqname')
dfkeep.to_csv('processed_data/screeningvar138_2000.csv', index=False)

