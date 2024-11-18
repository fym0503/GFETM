#! /bin/bash

python bin/scbasset_preprocess.py --ad_file data/downloads/buen_ad_sc.h5ad --input_fasta examples/hg19.fa --out_path all_data/buen18/processed_ratio
python scBasset/scbasset_train.py --input_folder scBasset/all_data/PBMC --out_path scBasset/scbasset_pbmc_replicate/ --epochs 1000 
