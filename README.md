# GFETM
**GFETM: Genome Foundation-based Embedded Topic Model for scATAC-seq Modeling**

Preprint Link: https://www.biorxiv.org/content/10.1101/2023.11.09.566403v1.full.pdf

RECOMB 2024 Conference version: https://dl.acm.org/doi/10.1007/978-1-0716-3989-4_20

![framework_gfetm](https://github.com/user-attachments/assets/6e090921-21d0-4089-a6fc-7b8db5fa14a2)

## Environment Configuration and Installation
```
git clone https://github.com/fym0503/GFETM.git
cd GFETM
conda create -n GFETM_env python=3.8.10
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation
We provided a sample dataset in Google Drive as instructed in [data](https://github.com/fym0503/GFETM/tree/main/data). To pre-process your own datasets, please follow the instructions in https://github.com/fym0503/GFETM/blob/main/scripts/general/data_preprocess.ipynb. To proceed with the preprocessing, please make sure your dataset has a .h5ad format with .var['chr','start','end'] indicating the chromosomes, start position and end positin of the peak coordinates.

## Tutorials
We provided a minimal tutorial at https://github.com/fym0503/GFETM/blob/main/scripts/tutorial_minimal.ipynb

## Reproducibility
We provided some scripts for replicating figures in our study at https://github.com/fym0503/GFETM/blob/main/scripts/

## Contact
The full paper is still under review. If you have any questions about the code, feel free to propose an issue or email at fanyimin.fym@link.cuhk.edu.hk
