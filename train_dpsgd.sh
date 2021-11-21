#!/bin/bash
#SBATCH --gres=gpu:p100:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out
#SBATCH --account=def-gckamath

module load python/3.8

source ../Pytorch/bin/activate

python train_dpsgd.py --model_name=cifar_net --batch_size=1024 --val_batch_size=1024 --mini_batch_size=1024 --lr=0.01 --optim=Adam --momentum=0.9 --noise_multiplier=0.0 --max_grad_norm=99999 --epochs=100 --data_path=/data --weight_decay=0 --lr_decay_factor=0.1 -lr_decay_epoch 30 60 80 --CheckPointPATH=./Checkpoints/Testresnet/ --No-DP
