import numpy as np
import anndata
topic_mixture = np.loadtxt("data/topic_mixture.txt")
adata=anndata.read_h5ad("data/adata.h5ad")

adata.obsm['topic_mixture'] = topic_mixture
diseases = np.unique(adata.obs['disease'])
disease_de_topics = []
for j in range(len(diseases)):
    positive_index = adata.obs['disease']==diseases[j]
    positive = np.mean(adata[positive_index].obsm['topic_mixture'],axis=0)
    negative_index = adata.obs['disease']!=diseases[j]
    negative = np.mean(adata[negative_index].obsm['topic_mixture'],axis=0)
    disease_de_topics.append(positive-negative)

# Statistical tests for the Differential topic identification for disease
'''
count = np.zeros((len(diseases),64))
embed = adata.obsm['topic_mixture']
for i in range(0,100002):
    index = np.arange(adata.shape[0])
    np.random.shuffle(index)
    adata.obs['disease_rand'] = np.array(adata.obs['disease'][index])
    for j in range(len(diseases)):
        positive_index = adata.obs['disease_rand']==diseases[j]
        positive = np.mean(embed[positive_index],axis=0)
        negative_index = adata.obs['disease_rand']!=diseases[j]
        negative = np.mean(embed[negative_index],axis=0)
        res = positive - negative
        count[j][res>disease_de_topics[j]] += 1
  
np.savetxt("data/count.txt")
'''

count = np.loadtxt("data/count_disease.txt")
count = (count + 1) / (100000 + 1)
count = count * 64 * 2
de_topics_all = []
for i in range(count.shape[0]):
    de_topics = []
    p_val = np.where(count[i]<0.01)[0]
    margin = np.where(disease_de_topics[i] > 0.2)[0]
    de_topics = set(list(p_val)).intersection(set(list(margin)))
    de_topics = list(de_topics)
    de_topics_all.append(de_topics)


np.savetxt("data/kidey_disease_p_val.txt",count)
np.savetxt("data/kidey_disease_margin.txt",np.stack(disease_de_topics))

