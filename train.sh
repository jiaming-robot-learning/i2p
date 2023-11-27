
# change this variable to match your machine.
N_GPU=4

python train.py --n-gpu-per-node $N_GPU --beta-max 0.3 --name i500_data

