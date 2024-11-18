import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scanpy as sc
import anndata
import torch
import sys
import pandas as pd

adata = anndata.read_h5ad("data/adata.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.tl.rank_genes_groups(adata, 'cell_type', method='t-test')


peak_topic = np.loadtxt("data/peak_by_topic.txt")

hvg_index = dict()
de_peaks = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10)
all_peaks = adata.var['feature_name'].index.tolist()
cell_names = de_peaks.columns.tolist()
for j in range(16):
    temp = []
    hvg_index[j] = temp
    for i in range(10):

        hvg_index[j].append(all_peaks.index(de_peaks.iloc[i][j]))

hvg_index_1 = []
for i in range(len(hvg_index)):
    for j in hvg_index[i]:
        if j not in hvg_index_1:
            hvg_index_1.append(j)

df1 = np.loadtxt("data/kidey_cell_type_p_val.txt")
df3 = np.loadtxt("data/kidey_cell_type_margin.txt")
index1 = np.where(df1<0.012)
list1 = [(index1[0][i],index1[1][i]) for i in range(index1[0].shape[0])]
index3 = np.where(df3>=1)

select_topic=[]
select_peak =[]
for i in range(16):
    to_add_topic = list(np.where(df3[i]>=1)[0])
    for j in to_add_topic:
        if j not in select_topic:
            select_topic.append(j)
    for j in hvg_index[i]:
        if j not in select_peak:
            select_peak.append(j)
select_topic = [9, 38, 55, 43, 10, 57, 19]
mat = peak_topic[list(select_peak)][:,list(select_topic)]

row_names = np.array(all_peaks)[select_peak]
select_id = list(np.arange(50)) + list(np.arange(110,136))
mat = mat[select_id]
for i in range(mat.shape[1]):
    mat[:,i] = 2* (mat[:,i] - np.min(mat[:,i]) )/ (np.max(mat[:,i])- np.min(mat[:,i])) -1

row_names = row_names[select_id]
col_names = select_topic
df = pd.DataFrame(mat, index=list(row_names), columns=col_names)
df.to_csv("data/peak_by_topic_panel.csv")
