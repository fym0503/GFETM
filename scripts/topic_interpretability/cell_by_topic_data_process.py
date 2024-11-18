import numpy as np
import anndata
import pandas as pd
adata = anndata.read_h5ad("data/adata.h5ad")
embed = np.loadtxt("data/topic_mixture.txt")
adata.obsm['X_embed'] = embed
index = np.arange(embed.shape[0])

adata_sample = adata[index].copy()
conditions = adata_sample.obs['disease']
cell_types = adata_sample.obs['cell_type']

map_cell_types = np.unique(cell_types)
maps_cell={}
count = 0
for i in np.unique(cell_types):
    maps_cell[i] = count
    count += 1

cell_types_int = []
for i in range(len(cell_types)):
    cell_types_int.append(maps_cell[cell_types[i]])
cell_types_int = np.array(cell_types_int)
rank_idx = []
for i in range(16):
    rank_idx.append([])
for i in range(len(cell_types_int)):
    if conditions[i] == 'normal':
        rank_idx[cell_types_int[i]].append(i)
    else:
        rank_idx[cell_types_int[i]].insert(0,i)
idx = []
for i in rank_idx:
    idx += i

dict1 = dict()
dict1['conditions'] = np.array(conditions[idx])
dict1['cell_type'] = cell_types[idx]
df=pd.DataFrame(dict1)
df.to_csv("data/cell_by_topic_metadata.csv")
embeds = adata_sample.obsm['X_embed']
df1 = pd.DataFrame(embeds[idx])
df1.to_csv("data/cell_by_topic_embeds.csv")
