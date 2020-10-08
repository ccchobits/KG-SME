#!/bin/bash
#SBATCH -J model_test
#SBATCH --gres=gpu:1
#SBATCH -p PV100q
#SBATCH -w node09

margin=(1.0 2.0)
dataset_name=WN18
bern=(False True)
epochs=120
batch_size=(512 1024 2048)
learning_rate=0.01
dim=64
lr_decay=1.8
norm=1

#python3 ../main/main.py --dim 64 --bs 512 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 1.0 --norm 1

#python3 ../main/main.py --gpu 3 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 100

#python3 ../main/main.py --gpu 3 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 100

#python3 ../main/main.py --gpu 3 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 120

#python3 ../main/main.py --gpu 3 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 120

#python3 ../main/main.py --gpu 3 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 10.0 --norm 1 --epochs 1000 --hidden 100

#python3 ../main/main.py --gpu 3 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 10.0 --norm 1 --epochs 1000 --hidden 100

#python3 ../main/main.py --gpu 3 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 10.0 --norm 1 --epochs 1000 --hidden 120

#python3 ../main/main.py --gpu 3 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 10.0 --norm 1 --epochs 1000 --hidden 120

#python3 ../main/main.py --gpu 3 --dim 30 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 48

#python3 ../main/main.py --gpu 3 --dim 128 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 196

#python3 ../main/main.py --gpu 3 --dim 256 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 300

#python3 ../main/main.py --gpu 3 --dim 512 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0 --norm 1 --epochs 1000 --hidden 600

#python3 ../main/main.py --gpu 3 --dim 512 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 10.0 --norm 1 --epochs 1000 --hidden 600

#python3 ../main/main.py --gpu 3 --dim 512 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 20.0 --norm 1 --epochs 1000 --hidden 600

#python3 ../main/main.py --gpu 3 --dim 512 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 4.0 --norm 1 --epochs 1000 --hidden 600

#python3 ../main/main.py --gpu 3 --dim 768 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 4.0 --norm 1 --epochs 1000 --hidden 900

#python3 ../main/main.py --gpu 3 --dim 768 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 900

#python3 ../main/main.py --gpu 3 --dim 768 --bs 2048 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 900

#python3 ../main/main.py --gpu 3 --dim 1024 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 1024

#python3 ../main/main.py --gpu 3 --dim 1024 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 1024 --reg 0.00001

#python3 ../main/main.py --gpu 3 --dim 1024 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 1024 --reg 0.00002

#python3 ../main/main.py --gpu 3 --dim 1280 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 1024 --reg 0.000

python3 ../main/main.py --gpu 3 --dim 1280 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 3.0 --norm 1 --epochs 1000 --hidden 1280 --reg 0.000
























#rm "slurm-${SLURM_JOB_ID}.out"
