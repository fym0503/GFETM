import torch
import torch.nn.functional as F 
import numpy as np 
from pathlib import Path
from GFETM.utils import nearest_neighbors, get_topic_coherence, get_topic_diversity
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from torch import nn, optim
from transformers import AutoModel,AutoConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.distributions as dist

class SCATAC_ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings=None, load_embeddings=True, fix_embeddings=True, enc_drop=0.5, batch_correction=True, cell_num = 2034, n_batches=2,checkpoint_path="dnabert"):
        super(SCATAC_ETM, self).__init__()

        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.batch_correction = batch_correction
        self.theta_act = self.get_activation(theta_act)
        self.sigmoid = nn.Sigmoid()
        self.update = 0
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
        config = AutoConfig.from_pretrained(checkpoint_path)
        self.peak_encoder = AutoModel.from_pretrained(checkpoint_path, config=config).to(device)
        if batch_correction:
            self.bc = nn.Parameter(torch.randn(n_batches, vocab_size))
        for p in self.peak_encoder.named_parameters():
            if "11" in p[0] or "10" in [0]:
                continue
            else:
                p[1].requires_grad=False

        print(vocab_size, " THE Vocabulary size is here ")
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, 128), 
                nn.ReLU(),
                nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        self.mu_q_theta = nn.Linear(128, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(128, num_topics, bias=True)

    
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self,peaks):
        """
        This generate the description as a defintion over words

        Returns:
            [type]: [description]
        """
        peak_embeds = self.forward_peak_encoder(peaks)
        logit = self.alphas(peak_embeds) 
        beta = F.softmax(logit, dim=0).transpose(1, 0) 
        return logit, beta

    def get_theta(self, normalized_bows):
        """
        getting the topic poportion for the document passed in the normalixe bow or tf-idf"""

        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta.clamp(-10,10))
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta
    def decode(self, theta, beta,batch_indices=None):
        """compute the probability of topic given the document which is equal to theta^T ** B
        Args:
            theta ([type]): [description]
            beta ([type]): [description]
        Returns:
            [type]: [description]
        """
        res = torch.mm(theta,beta.T)
        if self.batch_correction: #and self.epoch > 500:
            res += self.bc[batch_indices]
        almost_zeros = torch.full_like(res, 1e-30)
        results_without_zeros = res.add(almost_zeros)
        predictions = torch.log(F.softmax(results_without_zeros,dim=-1)+ 1e-30)
        return predictions

    def forward_joint(self, bows, normalized_bows, peak_feature, batch_indices=None,select_peaks=None, theta=None, aggregate=True):

        if theta is None:
            theta, kld_theta = self.get_theta(bows)
        else:
            kld_theta = None

        peak_feature = peak_feature.to(device)

        peak_embeds = self.encode_peak(peak_feature)
        temp2 = self.alphas(peak_embeds)
        preds = torch.mm(theta, temp2.T)
        
        if self.batch_correction: #and self.epoch > 500:
            preds += self.bc[batch_indices][:,select_peaks]
            
        output = torch.log(F.softmax(preds)+1e-30)
        loss_cell = -(output * normalized_bows).sum()

        return loss_cell, kld_theta, peak_embeds
    def encode_peak(self,seqs):
        x = self.peak_encoder(seqs)[0]
        x = torch.mean(x,dim=1)
        return x
    def forward_peak(self, seqs, peaks):
        x = self.peak_encoder(seqs)[0]
        x = torch.mean(x,dim=1)
        x = self.peak_fc(x)
        x = self.peak_clf(x)
        x = self.sigmoid(x)
        loss = nn.BCELoss()
        loss_item = loss(x, peaks)
        return loss_item
    
    def get_optimizer(self, args):
        if args.optimizer == 'adam':
            cell_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.lr)
            peak_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.lr)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(self.parameters(), lr=args.lr)
        self.cell_optimizer = cell_optimizer
        self.peak_optimizer = peak_optimizer
        return cell_optimizer, peak_optimizer

   

    def train_for_epoch(self, epoch, args, training_set_cell, peak_feature):
        self.train()
        acc_loss = 0
        acc_loss_cell = 0
        acc_loss_peak = 0
        acc_loss = 0
        self.epoch = epoch
    
        self.cell_optimizer.zero_grad()

        
        if True:
            for step, data_batch in enumerate(training_set_cell):
                select_peaks = torch.randint(0,peak_feature.shape[0], (256,))
                peak_input = peak_feature[select_peaks]

                selected_data = data_batch[0][:,select_peaks] 
                selected_data = selected_data / (torch.sum(selected_data,dim=0) + 1e-7)
                recon_loss, kld_theta, peak_embeds = self.forward_joint(data_batch[0].to(device), selected_data.to(device), peak_input, data_batch[1].to(device),select_peaks)
                if self.batch_correction: 
                    l2_reg = torch.norm(self.bc, p=1)  # L2 norm
                    total_loss = recon_loss +  0.00001 * kld_theta + 50 * l2_reg
                else:
                    total_loss = recon_loss +  0.00001 * kld_theta
                total_loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
                self.cell_optimizer.step()
                acc_loss += torch.sum(recon_loss).item()
                self.cell_optimizer.zero_grad()
                self.zero_grad()
        

            
        print('*'*100)
        print('Epoch----->Train step {} ..  Rec_loss: {} '.format(
                epoch,  acc_loss))
        print('*'*100)
        
        
        
        return acc_loss
   
