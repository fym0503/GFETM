train_data=$1
test_data=$2
save_dir=$3
seed=$4
python peakvi.py --train_data ${train_data} --test_data ${test_data} --save_dir ${save_dir} --seed ${seed}