
import pandas as pd
import numpy as np
import scipy.stats as ss

from matplotlib import pyplot as plt 
import seaborn as sns
sns.set() 


sns.set_style("whitegrid")
sns.set(font = 'sans-serif', style = 'whitegrid')


## 距离阵热图
dfcorrdist = pd.read_csv("processed_data/corrdist.csv")

dfdist = pd.read_pickle("processed_data/dfdist.zip")
dfdistmean = pd.DataFrame(dfdist.mean())
dfdistmean.reset_index(drop=False, inplace=True)
dfdistmean.columns = ['varname', 'value']

dfplddt = pd.read_pickle("processed_data/dfplddt.zip")
dfplddtmean = pd.DataFrame(dfplddt.mean()) #没有不需要计算的列
dfplddtmean.reset_index(drop=False, inplace=True)
dfplddtmean.columns = ['varname', 'value']

dfplddt0 = dfplddtmean.copy()
dfplddt0['headfont'] = dfplddt0.varname.apply(lambda x : x[0])
dfplddt0['site1'] = dfplddt0.varname.apply(lambda x : x[1:x.find("_")])
dfplddt0['site2'] = dfplddt0.varname.apply(lambda x : x[x.find("_")+1:])
dfplddt0 = dfplddt0.loc[dfplddt0.site1 == dfplddt0.site2,:]
dfplddt0.value = np.sqrt(dfplddt0.value)

dfcorrangle = pd.read_csv("processed_data/corrangle.csv")
dfpsi = dfcorrangle.loc[dfcorrangle.varname.apply(lambda x : 'psi' in x) , :].copy()
dfpsi['cnt'] = dfpsi.varname.apply(lambda x : int(x[x.find("_")+1:]))
dfpsi.sort_values("cnt", inplace=True)

dfphi = dfcorrangle.loc[dfcorrangle.varname.apply(lambda x : 'phi' in x) , :].copy()
dfphi['cnt'] = dfphi.varname.apply(lambda x : int(x[x.find("_")+1:]))
dfphi.sort_values("cnt", inplace=True)


def df2mtx(dfmtx, MSA_len = 355, headlen = 1, half = True): # 将长形df转换回矩阵格式mtx

    dfmtx['headfont'] = dfmtx.varname.apply(lambda x : x[0 : headlen])
    dfmtx['site1'] = dfmtx.varname.apply(lambda x : x[headlen : x.find("_")])
    dfmtx['site2'] = dfmtx.varname.apply(lambda x : x[x.find("_")+1:])

    mtx = np.full((MSA_len, MSA_len), np.nan)

    for i in range(len(dfmtx)):
        mtx[int(dfmtx.loc[i, 'site1']), int(dfmtx.loc[i, 'site2'])] = dfmtx.loc[i, 'value']
        if half:
            mtx[int(dfmtx.loc[i, 'site2']), int(dfmtx.loc[i, 'site1'])] = dfmtx.loc[i, 'value']

    namelst = [str(i) for i in range(1,len(mtx)+1)]
    dfmtx = pd.DataFrame(mtx)
    dfmtx.columns = namelst
    dfmtx.index = namelst

    return dfmtx

distmtx = df2mtx(dfdistmean)
dfcorrdist['value'] = dfcorrdist['abscorr']
abscorrdistmtx = df2mtx(dfcorrdist)
dfcorrdist['value'] = dfcorrdist['corr']
corrdistmtx = df2mtx(dfcorrdist)
plddtmtx = df2mtx(dfplddtmean)

# dfcorrdist.loc[dfcorrdist.p > 0.05, 'value'] = 0 # 归零无统计意义的r
corrdistmtx = df2mtx(dfcorrdist)

mtx = pd.DataFrame(corrdistmtx)
mtx.iloc[25:280, 25:280]

################# 绘图代码

plt.figure(figsize = (15,15)) 
lst = []
for i in range(1,356):
    if i % 10 == 0:
        lst.append(i)

ax = sns.heatmap(plddtmtx, cmap = sns.color_palette("Blues", 100), square = True);  # 去掉 center=0 颜色区分更明显
plt.xticks(lst, lst); plt.yticks(lst, lst); plt.title("pLDDT"); plt.show()

ax = sns.heatmap(distmtx, cmap = sns.color_palette("Blues", 100), square = True);  # 去掉 center=0 颜色区分更明显
plt.xticks(lst, lst); plt.yticks(lst, lst); plt.title("Distance"); plt.show()

ax = sns.heatmap(abscorrdistmtx, cmap = sns.color_palette("Blues", 100), square = True, center=0);
plt.xticks(lst, lst); plt.yticks(lst, lst); plt.title("Distance Abs(Pearson Corrleation)"); plt.show()

ax = sns.heatmap(corrdistmtx, cmap = sns.color_palette("RdBu", 100), square = True, center=0);
plt.xticks(lst, lst); plt.yticks(lst, lst); plt.title("Distance Pearson Corrleation"); plt.show()


# sns.barplot(x = dfpsi.cnt, y = dfpsi.abscorr); plt.show()
plt.bar(dfphi.cnt, dfphi.abscorr); plt.xticks(lst, lst); plt.title("PHI Abs(Pearson Corrleation)"); plt.show()

plt.bar(dfpsi.cnt, dfpsi.abscorr); plt.xticks(lst, lst); plt.title("PSI Abs(Pearson Corrleation)"); plt.show()

plt.bar(dfplddt0.site1, dfplddt0.value); plt.xticks(lst, lst); plt.title("pLDDT"); plt.show()





# 查看绘图元素大小现有设定
sns.plotting_context()



dfres = pd.read_csv("processed_data/seqbrowse.csv")

axn = sns.histplot(data = dfres, x = 'maxage')
axn.figure.set_size_inches(6.4,4.8)
# sns.despine(top = False, right = False, left = False, bottom = False)
# ns.despine(fig = fig)

plt.xlabel('Maximum lifespan (MLS)')
plt.ylabel("Frequency")
plt.show()



dfres = pd.read_csv("pred/pred_t33.csv")

fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

sns.scatterplot(data = dfres, x = 'y', y = 'y_pred')
plt.xlabel('Maximum lifespan (MLS)')
plt.ylabel("Predicted MLS")
plt.show()



# entropy和mse

dfres = pd.read_csv("processed_data/mse275nohuman.csv")

dfres.sort_values("entropy2", inplace = True)
dfres.reset_index(drop = True, inplace = True)
dfres['pct'] = 100 * (dfres.index + 1 ) / 250


fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

sns.lineplot(data = dfres, x = 'entropy2', y = 'pct', ci = None)
plt.axvline(x=0.4, ymin=0.04,  c='darkgrey', ls='dashed', lw=1.5)

plt.text(0.4, 100, " >0.4", c = 'grey')
plt.xlabel('Entropy')
plt.ylabel("Cumulative % of site")
plt.show()



dfres.sort_values("mse", inplace = True)
dfres.reset_index(drop = True, inplace = True)
dfres['pct'] = 100 * (dfres.index + 1 ) / 250


fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

sns.lineplot(data = dfres, x = 'mse', y = 'pct', ci = None)
plt.axvline(x=335, ymin=0.04,  c='darkgrey', ls='dashed', lw=1.5)
plt.ylim(-4.58, 104.98)
plt.text(325, 100, "<335", c = 'grey')
plt.xlabel('Mean Square Error (MSE)')
plt.ylabel("Cumulative % of site")
plt.show()


ymin, ymax = plt.ylim()
print(ymin, ymax)


######## 不同subset的箱图
dfres = pd.read_csv("processed_data/boxplot.csv")
dfres.loc[dfres.cls ==103, 'cls'] = 104

fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

plt.axhline(y=0, c='grey', ls='dashed', lw=1.5)
sns.violinplot(y = dfres.lift, x = dfres.cls, palette='muted')

plt.xlabel('# of sites in subset')
plt.ylabel("Predicted MLS increase (PMI)")
plt.show()


dfres = dfres.loc[dfres.cls == 159,:]
sns.violinplot(y = dfres.lift)


# 5cv箱图
dfres = pd.read_pickle("processed_data/dfres_159.zip").loc[:, ["pred0",'pred1','pred2', 'pred3','pred4']]
dfres.columns = ["Model0",'Model1','Model2', 'Model3','Model4']

dfs = dfres.stack()

df2 = dfs.reset_index()
df2.columns = ["a",'b','pred']

fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

sns.violinplot(y = df2.pred, x = df2.b)
plt.xlabel('# of fold for T33 model')
plt.ylabel("Predicted MLS")
plt.show()



########## subset如何划分的线图
# 双轴线图
dfres = pd.read_excel("memo.xlsx", sheet_name="sitegph")

sns.lineplot(data = dfres, y = 'rmean', x = "ranknew", ci = None)
plt.axvline(x=104, ymin=0.0,  c='darkgrey', ls='dashed', lw=1.5)
plt.axvline(x=74, ymin=0.0,  c='darkgrey', ls='dashed', lw=1.5)
plt.axvline(x=50, ymin=0.0,  c='darkgrey', ls='dashed', lw=1.5)
plt.axvline(x=37, ymin=0.0,  c='darkgrey', ls='dashed', lw=1.5)

plt.text(104, 0.121, "104", c = 'grey')
plt.text(74, 0.121, "74", c = 'grey')
plt.text(50, 0.121, "50", c = 'grey')
plt.text(37, 0.121, "37", c = 'grey')
         
plt.xlabel('# of sites left in subset')
plt.ylabel("Average PMI of cumulative reducing variation sites")

# 设置第二y轴对应的图形
ax2 = plt.gca().twinx()
sns.lineplot(data = dfres, y = 'diff', x = "ranknew", ci = None, color = 'cadetblue',  ax = ax2)
plt.ylim(-0.005, 0.005)

plt.ylabel("Mean change of PMI when site replaced back")
plt.show()




########### 位点比例绘图
dfres = pd.read_pickle("df569sitepct.zip")
dfres = dfres.loc[dfres.cls == '159',['pct37', 'pct50','pct74','pct104','pct159']]
dfs = dfres.stack()
df2 = dfs.reset_index()
df2.columns = ["a",'subset','pct']
df2.subset.replace(['pct37', 'pct50', 'pct74', 'pct104'], ['1-37', '38-50', '51-74', '75-104'], inplace = True)

fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

plt.axhline(y=37/104, xmin = 0.05, xmax = 0.2, c='grey', ls='dashed', lw=2); plt.text(0.3, 37/104-0.01, "35.6%", c = 'grey')
plt.axhline(y=(50-37)/104, xmin = 0.3, xmax = 0.45, c='grey', ls='dashed', lw=2); plt.text(1.3, (50-37)/104-0.01, "12.5%", c = 'grey')
plt.axhline(y=(74-50)/104, xmin = 0.55, xmax = 0.7, c='grey', ls='dashed', lw=2); plt.text(2.3, (74-50)/104-0.01, "23.1%", c = 'grey')
plt.axhline(y=(104-74)/104, xmin = 0.8, xmax = 0.95, c='grey', ls='dashed', lw=2); plt.text(3.3, (104-74)/104-0.01, "28.8%", c = 'grey')

sns.violinplot(x = df2.subset, y = df2.pct)
plt.xlabel('Subset range the site belongs')
plt.ylabel("% in all mutated sites")
plt.show()


########### 37的四区段比较
########### 37的四区段比较，去掉raw
dfres = pd.read_pickle("processed_data/dfres37_6grp.zip")
dfres0 = pd.read_pickle('processed_data/dftopseq569.zip')
dfres0['grp'] = 'Raw'

dfres = pd.concat([dfres0, dfres], ignore_index=True)
dfres = dfres.query("grp != 'nadother' & grp != 'a2' & grp != 'Raw'")
dfres.r = dfres.r * 100

dfres.grp.replace(['Other', 'a1', 'b1', 'nad'], ['Other', 'Alpha1-helix', 'Beta1-sheet', 'NAD+ binding area'], inplace = True)

from pandas.api.types import CategoricalDtype

cat_size_order = CategoricalDtype(
    ['NAD+ binding area', 'Beta1-sheet', 'Alpha1-helix', 'Other'], 
    ordered=True
)

dfres['grp'] = dfres['grp'].astype(cat_size_order)
# dfres.grp.replace(['alpha1'], ['α1'], inplace = True)

fig = plt.figure()
ax1 = plt.gca()
ax1.spines['left'].set_color('darkgrey')
ax1.spines['right'].set_color('darkgrey')
ax1.spines['top'].set_color('darkgrey')
ax1.spines['bottom'].set_color('darkgrey')

plt.axhline(y=6.668278, c='grey', ls='dashed', lw=1.5)
plt.text(-0.49, 7, "Raw mean PMI = 6.67%", c = 'grey')
sns.violinplot(y = dfres.r, x = dfres.grp, palette=['magenta', 'orange', 'yellow', 'limegreen'])

plt.xlabel('Spatial parts of subset 37 been replaced back')
plt.ylabel("Predicted MLS Increase (PMI)")
plt.show()

