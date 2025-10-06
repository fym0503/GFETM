import torch
import numpy as np
import scipy
import anndata
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, TensorDataset


class SparseTensorDataset(Dataset):
    def __init__(self, sparse_features, indices):
        self.sparse_features = sparse_features
        self.indices = indices

    def __len__(self):
        return self.sparse_features.shape[0]

    def __getitem__(self, idx):
        row = self.sparse_features[idx].to_dense()
        return row, self.indices[idx]


class SparsePeakDataset(Dataset):
    def __init__(self, features, sparse_matrix, indices):
        self.features = features
        self.sparse_matrix = sparse_matrix
        self.indices = indices

    def __len__(self):
        return self.sparse_matrix.shape[1]

    def __getitem__(self, idx):
        col = self.sparse_matrix[:, idx].to_dense()
        return self.features[idx], col, self.indices[idx]


def load_and_cache_examples(args, split='train', evaluate=False):
    data_path = args.data_path if split == 'train' else args.full_data_path
    adata = anndata.read_h5ad(data_path)
    
    count_matrix = np.ceil(adata.X.toarray() / 2) if args.fragment == 1 else adata.X.toarray()
    
    tfidf_vectorizer = TfidfTransformer()
    tfidf_transformed = tfidf_vectorizer.fit_transform(count_matrix)
    normalized_matrix = normalize(tfidf_transformed, axis=1, norm='l1')
    
    cell_indices = torch.arange(count_matrix.shape[0], dtype=torch.long)
    peak_indices = torch.arange(count_matrix.shape[1], dtype=torch.long)
    peak_features = torch.LongTensor(anndata.read_h5ad(args.seq_path).X)

    if args.use_sparse:
        sparse_counts = scipy.sparse.coo_matrix(count_matrix) if not scipy.sparse.issparse(count_matrix) else count_matrix.tocoo()
        peak_cell_matrix = torch.sparse_coo_tensor(
            torch.tensor([sparse_counts.row, sparse_counts.col], dtype=torch.long),
            torch.tensor(sparse_counts.data, dtype=torch.float),
            sparse_counts.shape
        )
        
        sparse_normalized = normalized_matrix.tocoo()
        normalized_sparse_tensor = torch.sparse_coo_tensor(
            torch.tensor([sparse_normalized.row, sparse_normalized.col], dtype=torch.long),
            torch.tensor(sparse_normalized.data, dtype=torch.float),
            sparse_normalized.shape
        )
        
        cell_dataset = SparseTensorDataset(normalized_sparse_tensor, cell_indices)
        peak_dataset = SparsePeakDataset(peak_features, peak_cell_matrix, peak_indices)
    else:
        peak_cell_matrix = torch.clamp(torch.tensor(count_matrix, dtype=torch.float), max=1.0)
        normalized_dense = torch.tensor(normalized_matrix.toarray() if hasattr(normalized_matrix, 'toarray') else normalized_matrix, dtype=torch.float)
        
        cell_dataset = TensorDataset(normalized_dense, cell_indices)
        peak_dataset = TensorDataset(peak_features, peak_cell_matrix.T, peak_indices)
    
    gc.collect()
    return cell_dataset, peak_dataset, peak_features, peak_cell_matrix