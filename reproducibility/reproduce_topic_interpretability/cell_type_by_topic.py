import numpy as np
import anndata
import pandas as pd
topic = np.arange(64)
adata = anndata.read_h5ad("data/adata.h5ad")
cell_type = np.unique(adata.obs['cell_type'])
disease = np.unique(adata.obs['disease'])
p_val_cell_type = np.loadtxt("data/kidey_cell_type_p_val.txt")
p_val_disease = np.loadtxt("data/kidey_disease_p_val.txt")

margin_cell_type = np.loadtxt("data/kidey_cell_type_margin.txt")
margin_disease = np.loadtxt("data/kidey_disease_margin.txt")
map_cell_types = np.unique(cell_type)
maps_cell={}
count = 0
for i in np.unique(cell_type):
    maps_cell[i] = count
    count += 1
    maps_cell[i] = i
select_topics = []
# df1 = pd.read_csv("kidney_p_val_cell_type.csv").values[:,1:]
# df2 = pd.read_csv("kidney_p_val_disease.csv").values[:,1:]
# df3 = pd.read_csv("kidney_margin_cell_type.csv").values[:,1:]
# df4 = pd.read_csv("kidney_margin_disease.csv").values[:,1:]
df1 = np.loadtxt("data/kidey_cell_type_p_val.txt")
df2 = np.loadtxt("data/kidey_disease_p_val.txt")
df3 = np.loadtxt("data/kidey_cell_type_margin.txt")
df4 = np.loadtxt("data/kidey_disease_margin.txt")

index1 = np.where(df1<0.012)
list1 = [(index1[0][i],index1[1][i]) for i in range(index1[0].shape[0])]
index3 = np.where(df3>0.5)
list3 = [(index3[0][i],index3[1][i]) for i in range(index3[0].shape[0])]

target_list = list(set(list1).intersection(set(list3)))
for i in target_list:
    select_topics.append(i[1])
index2 = np.where(df2<0.012)
list2 = [(index2[0][i],index2[1][i]) for i in range(index2[0].shape[0])]
index4 = np.where(df4>0.5)
list4 = [(index4[0][i],index4[1][i]) for i in range(index4[0].shape[0])]

target_list = list(set(list2).intersection(set(list4)))
for i in target_list:
    select_topics.append(i[1]) 
topics = list(set(select_topics))
# import pdb;pdb.set_trace()

dict_df = dict()
for i in range(p_val_cell_type.shape[0]):
    dict_df[maps_cell[cell_type[i]]] = p_val_cell_type[i]
pd.DataFrame(dict_df).transpose().to_csv("data/kidney_p_val_cell_type.csv")

dict_df = dict()
for i in range(p_val_disease.shape[0]):
    dict_df[disease[i]] = p_val_disease[i]
pd.DataFrame(dict_df).transpose().to_csv("data/kidney_p_val_disease.csv")
dict_df['diabetes'] = dict_df['type 2 diabetes mellitus']
del dict_df['type 2 diabetes mellitus']
dict_df = dict()
for i in range(margin_cell_type.shape[0]):
    dict_df[maps_cell[cell_type[i]]] = margin_cell_type[i]
pd.DataFrame(dict_df).transpose().to_csv("data/kidney_margin_cell_type.csv")

dict_df = dict()
for i in range(margin_disease.shape[0]):
    dict_df[disease[i]] = margin_disease[i]
dict_df['diabetes'] = dict_df['type 2 diabetes mellitus']
del dict_df['type 2 diabetes mellitus']
pd.DataFrame(dict_df).transpose().to_csv("data/kidney_margin_disease.csv")
#     select_topics += list(np.where(p_val_cell_type[i]<0.012)[0])
#     print(set(select_topics))
# select_topics = list(set(select_topics))
