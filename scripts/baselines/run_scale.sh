# git clone https://github.com/jsxlei/SCALE.git
# cd SCALE
# pip install -e .
input=$1
outdir=$2
python SCALE/SCALE.py -d ${input} --impute --outdir ${outdir}