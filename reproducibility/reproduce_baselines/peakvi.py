import anndata
import scvi
import scanpy as sc
import h5py
import scipy
import pandas as pd
import argparse
import numpy as np
import csv
from sklearn.metrics.cluster import adjusted_rand_score
import time
from sklearn.metrics.cluster import adjusted_mutual_info_score
scvi.settings.verbosity = 40
parser = argparse.ArgumentParser(description='The Embedded Topic Model')

parser.add_argument('--train_data', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--test_data', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--save_dir', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--seed', type=int, default=1000, help='directory containing data')

args = parser.parse_args()

scvi.settings.seed = args.seed

train_data = anndata.read_h5ad(args.train_data)
test_data = anndata.read_h5ad(args.test_data)

scvi.model.PEAKVI.setup_anndata(train_data)
pvi = scvi.model.PEAKVI(train_data)
pvi.train()

pvi.save(args.save_dir, overwrite=True)
pvi = scvi.model.PEAKVI.load(args.save_dir, test_data)
proj = pvi.get_latent_representation()

test_data.obsm['projection'] = proj
np.savetxt(args.save_dir + "/embeds.txt",proj)
