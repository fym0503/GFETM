import pandas as pd
import numpy as np
import scanpy as sc
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
import anndata
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
import sklearn

def getNClusters(adata,n_cluster,range_min=0,range_max=3,max_steps=20):
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata,resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()
        
        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            return(this_resolution, adata)
        this_step += 1
    
    print('Cannot find the number of clusters')
    print('Clustering solution from last iteration is used:' + str(this_clusters) + ' at resolution ' + str(this_resolution))

def shared_neighbor_label(ad,nn,label_obs_idx=1):
    sc.pp.neighbors(ad, n_neighbors=nn+1, use_rep='projection') # 100 nearest neighbor of each cell
    m_ATAC_neighbors = [i.indices for i in ad.obsp['distances']] # neighbor idx for each cell
    neighbor_label = ad.obs.iloc[np.concatenate(m_ATAC_neighbors, axis=0), label_obs_idx] # label for all neighbors
    cell_label = ad.obs.iloc[np.repeat(np.arange(len(m_ATAC_neighbors)), [len(j) for j in m_ATAC_neighbors]), label_obs_idx] # label for all cells
    n_shared = (neighbor_label.values==cell_label.values).sum() / len(m_ATAC_neighbors)
    return n_shared / nn


def eval(adata,proj):
    adata.obsm['projection'] = proj
    sc.pp.neighbors(adata, n_neighbors=15,use_rep='projection')
    getNClusters(adata,n_cluster=10)
    louvain = []
    for j in range(len(adata.obs['louvain'])):
        louvain.append(int(adata.obs['louvain'][j]))
    adata.obs['louvain'] = pd.Series(louvain,index=adata.obs.index).astype('category')
    ari = adjusted_rand_score(adata.obs['label'], adata.obs['louvain'])
    homogenity = sklearn.metrics.homogeneity_score(adata.obs['label'], adata.obs['louvain'])
    admis = adjusted_mutual_info_score(adata.obs['label'], adata.obs['louvain'],average_method='arithmetic')
    sil = silhouette_score(adata.obsm['projection'], adata.obs['label'])
    adata.uns['metrics']['ari'].append(ari)
    adata.uns['metrics']['homogenity'].append(homogenity)
    adata.uns['metrics']['admis'].append(admis)
    adata.uns['metrics']['sil'].append(sil)

    return {"ari":ari, }
