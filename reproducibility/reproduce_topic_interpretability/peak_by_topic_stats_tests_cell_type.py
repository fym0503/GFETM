import numpy as np
import anndata
topic_mixture = np.loadtxt("data/topic_mixture.txt")
adata=anndata.read_h5ad("data/adata.h5ad")
adata.obsm['topic_mixture'] = topic_mixture
cell_types = np.unique(adata.obs['cell_type'])
cell_type_de_topics = []
for j in range(len(cell_types)):
    positive_index = adata.obs['cell_type']==cell_types[j]
    positive = np.mean(adata[positive_index].obsm['topic_mixture'],axis=0)
    negative_index = adata.obs['cell_type']!=cell_types[j]
    negative = np.mean(adata[negative_index].obsm['topic_mixture'],axis=0)
    cell_type_de_topics.append(positive-negative)

# Statistical tests for the Differential topic identification for cell type
'''
count = np.zeros((len(cell_types),64))
embed = adata.obsm['topic_mixture']
for i in range(0,100002):
    index = np.arange(adata.shape[0])
    np.random.shuffle(index)
    adata.obs['cell_type_rand'] = np.array(adata.obs['cell_type'][index])
    for j in range(len(cell_types)):
        positive_index = adata.obs['cell_type_rand']==cell_types[j]
        positive = np.mean(embed[positive_index],axis=0)
        negative_index = adata.obs['cell_type_rand']!=cell_types[j]
        negative = np.mean(embed[negative_index],axis=0)
        res = positive - negative
        count[j][res>cell_type_de_topics[j]] += 1
    print(i)
np.savetxt("data/count_cell_type.txt",count)
'''
count = np.loadtxt("data/count_cell_type.txt")
count = (count + 1) / (100000 + 1)
count = count * 64 * 16
de_topics_all = []
for i in range(count.shape[0]):
    de_topics = []
    p_val = np.where(count[i]<0.011)[0]
    margin = np.where(cell_type_de_topics[i] > 0.8)[0]
    de_topics = set(list(p_val)).intersection(set(list(margin)))
    de_topics = list(de_topics)
    de_topics_all.append(de_topics)

np.savetxt("data/kidey_cell_type_p_val.txt",count)
np.savetxt("data/kidey_cell_type_margin.txt",np.stack(cell_type_de_topics))
