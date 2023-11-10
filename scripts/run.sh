#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=ctb-liyue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=125G

python main.py --mode train  --data_path ../data/buen18.h5ad --seq_path ../data/buen18_seq.h5 --checkpoint_path ../gfm_checkpoint/6-new-12w-0 --enc_drop 0.1 --num_topics 24 --seed 4 --epochs 8000 --emb_size 768 --rho_size 768 --t_hidden_size 1000 --output_path outputs/
