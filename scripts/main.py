#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfTransformer



from pathlib import Path
import tracemalloc
from torch.utils.data import RandomSampler, SequentialSampler
import anndata
from torch.utils.data import DataLoader, TensorDataset
import gc
from GFETM.model import SCATAC_ETM
import GFETM.eval
from sklearn.preprocessing import normalize
import torch.distributions as dist

parser = argparse.ArgumentParser(description='GFETTM')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--seq_path', type=str, default='data/20ng', help='directory containing data')

parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=3000, help='input batch size for training')
parser.add_argument('--output_path', type=str, default='./results', help='path to save results')
parser.add_argument('--resume_ckpt', type=str, default=None, help='path to save results')
parser.add_argument('--batch_correction', type=bool, default=False, help='path to save results')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=32, help='number of topics')
parser.add_argument('--rho_size', type=int, default=32, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=32, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=1000, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--checkpoint_path', type=str, default='dnabert', help='')

parser.add_argument('--load_embeddings', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--fix_embeddings', type=int, default=0, help='whether to fix rho or train it')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--log_interval', type=int, default=5, help='when to log training')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def load_and_cache_examples(args,split='train', evaluate=False):
  
    ad = anndata.read_h5ad(args.data_path)
    m = ad.X.toarray()#[0:200,0:500]
    train_dataset = np.array(m)
    peak_cell_matrix = torch.tensor(train_dataset,dtype=torch.float)
    peak_cell_matrix[peak_cell_matrix > 1] =1.0
    vectorizer = TfidfTransformer()
    vectorizer.fit(train_dataset)
    X = vectorizer.transform(train_dataset).toarray()
    X1 = normalize(X, axis=1, norm='l1')
    indices_cell = torch.tensor(np.arange(m.shape[0]),dtype=torch.long)
    indices_peak = torch.tensor(np.arange(m.shape[1]),dtype=torch.long)
    cached_features_file_train = args.seq_path
    feature = torch.LongTensor(anndata.read_h5ad(cached_features_file_train).X)
    dataset_cell = TensorDataset(torch.tensor(X1,dtype=torch.float),indices_cell)
    dataset_peak = TensorDataset(feature,peak_cell_matrix.T, indices_peak)
    gc.collect()
    return dataset_cell, dataset_peak, feature, peak_cell_matrix

train_dataset_cell, train_dataset_peak, peak_feature, peak_cell_matrix = load_and_cache_examples(args, 'train',False)
valid_dataset = load_and_cache_examples(args, 'train',True)

print(len(train_dataset_cell))


vocab_size = train_dataset_cell[0][0].shape
args.vocab_size = vocab_size[0]
args.num_docs_train = len(train_dataset_cell)
args.num_docs_valid = len(train_dataset_cell)

embeddings = None
if args.load_embeddings:
    embeddings = np.loadtxt(args.emb_path)
    embeddings = torch.tensor(embeddings).to(device)


print('=*'*100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = Path.cwd().joinpath(args.save_path, 
        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.load_embeddings, args.fix_embeddings))

if args.resume_ckpt is not None:
    model = torch.load(args.resume_ckpt).to(device)
    epochs_exist = int(args.resume_ckpt.split("_")[-1])
else:
    model = SCATAC_ETM(args.num_topics, 
                args.vocab_size, 
                args.t_hidden_size, 
                args.rho_size, 
                args.emb_size, 
                args.theta_act, 
                embeddings, 
                args.load_embeddings, 
                args.fix_embeddings,
                args.enc_drop,
                batch_correction = args.batch_correction,
                cell_num = len(train_dataset_cell),
                checkpoint_path=args.checkpoint_path,
                
                ).to(device)
    print('model: {}'.format(model))
    epochs_exist = 0
cell_optimizer, peak_optimizer = model.get_optimizer(args)

os.makedirs(args.output_path,exist_ok=True)
tracemalloc.start()
adata = anndata.read_h5ad(args.data_path)
adata.uns['metrics'] = {'ari':[],'admis':[],"homogenity":[],"sil":[]}
if args.mode == 'train':
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    print('\n')
    print('Visualizing model quality before training...', args.epochs)
    print('\n')
    train_sampler_cell = RandomSampler(train_dataset_cell)
    train_dataloader_cell = DataLoader(train_dataset_cell, sampler=train_sampler_cell, batch_size=args.batch_size)
    train_sampler_peak = RandomSampler(train_dataset_peak)
    train_dataloader_peak = DataLoader(train_dataset_peak,sampler=train_sampler_peak, batch_size=512)
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs_exist, args.epochs):
        print("I am training for epoch", epoch)
        train_loss = model.train_for_epoch(epoch, args, train_dataloader_cell,  peak_feature)

        train_loss_list.append(train_loss)
        if epoch >= 4000:
            lr = cell_optimizer.param_groups[0]['lr']

            lr = lr * np.exp(-6e-5)
            cell_optimizer.param_groups[0]['lr'] = lr       
        if epoch % args.log_interval == 0:
            test_dataset,_,_,_ = load_and_cache_examples(args, 'all',False)
            eval_sampler = SequentialSampler(test_dataset)
            eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=len(test_dataset))
            res = []
            model.eval()
            projs = []
            for i in eval_dataloader:
                projs.append(model.encode(i[0].to(device))[0].detach().cpu().numpy())
            proj = np.concatenate(projs)
            GFETM.eval.eval(adata,proj)
            if np.max(adata.uns['metrics']['ari']) == adata.uns['metrics']['ari'][-1]:
                adata.obsm['projection'] = proj
                torch.save(model, args.output_path + "/best_checkpoint" )
            adata.write(args.output_path + "/adata.h5ad")
                # np.savetxt(args.output_path + "/" + str(epoch) + "_out.txt",t.detach().cpu().numpy())
    
