import numpy as np
import torch 
from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split
from gensim.models.fasttext import FastText as FT_gensim
from tqdm import tqdm
import scanpy as sc
from cProfile import label
import pandas as pd
import numpy as np
import scanpy as sc
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import anndata
from sklearn.metrics.cluster import adjusted_mutual_info_score
# import tensorflow as tf
import torch
def read_mat_file(key, path):
    """
    read the preprocess mat file whose key and and path are passed as parameters

    Args:
        key ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    term_path = Path().cwd().joinpath('data', 'preprocess', path)
    doc = loadmat(term_path)[key]
    return doc

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
    

def split_train_test_matrix(dataset):
    """Split the dataset into the train set , the validation and the test set

    Args:
        dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=1)
    X_test_1, X_test_2 = train_test_split(X_test, test_size=0.5, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test_1, X_test_2

def get_data(doc_terms_file_name="tf_idf_doc_terms_matrix", terms_filename="tf_idf_terms"):
    """read the data and return the vocabulary as well as the train, test and validation tests

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    doc_term_matrix = read_mat_file("doc_terms_matrix", doc_terms_file_name)
    terms = read_mat_file("terms", terms_filename)
    vocab = terms
    train, validation, test_1, test_2 = split_train_test_matrix(doc_term_matrix)

    return vocab, train, validation, test_1, test_2

def get_batch(doc_terms_matrix, indices, device):
    """
    get the a sample of the given indices 

    Basically get the given indices from the dataset

    Args:
        doc_terms_matrix ([type]): the document term matrix
        indices ([type]):  numpy array 
        vocab_size ([type]): [description]

    Returns:
        [numpy arayy ]: a numpy array with the data passed as parameter
    """
    data_batch = doc_terms_matrix[indices, :]
    data_batch = torch.from_numpy(data_batch.toarray()).float().to(device)
    return data_batch


def read_embedding_matrix(vocab, device,  load_trainned=True):
    """
    read the embedding  matrix passed as parameter and return it as an vocabulary of each word 
    with the corresponding embeddings

    Args:
        path ([type]): [description]

    # we need to use tensorflow embedding lookup heer
    """
    model_path = Path.home().joinpath("Projects", 
                                    "Personal", 
                                    "balobi_nini", 
                                    'models', 
                                    'embeddings_one_gram_fast_tweets_only').__str__()
    embeddings_path = Path().cwd().joinpath('data', 'preprocess', "embedding_matrix.npy")

    if load_trainned:
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
    else:
        model_gensim = FT_gensim.load(model_path)
        vectorized_get_embeddings = np.vectorize(model_gensim.wv.get_vector)
        embeddings_matrix = np.zeros(shape=(len(vocab),50)) #should put the embeding size as a vector
        print("starting getting the word embeddings ++++ ")
        vocab = vocab.ravel()
        for index, word in tqdm(enumerate(vocab)):
            vector = model_gensim.wv.get_vector(word)
            embeddings_matrix[index] = vector
        print("done getting the word embeddings ")
        with open(embeddings_path, 'wb') as file_path:
            np.save(file_path, embeddings_matrix)

    embeddings = torch.from_numpy(embeddings_matrix).to(device)
    return embeddings
