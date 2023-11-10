#!/bin/bash

# params=(0.02 0.04 0.06 0.08 0.1) # gw radius
# params=(0.0 0.03 0.06 0.09 0.12) # unbalanced weight
# params=(0.1 0.3 0.5 0.7 0.9) # alpha
# params=(0.1 0.2 0.3 0.4 0.5) # rho
# params=(0.02 0.04 0.06 0.08 0.1) # training eps
params=(11 17 22 28 33) # number of classes
seed=0
gpu=0

for i in ${!params[@]}; do
	# python3 train.py -d FSeval -ac all -c ${params[$i]} -ne 20 -g 1 --seed 0 --group sens_K --rho 0.1 -ut 0.06 -at 0.3 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.04 --wandb
	# python3 train.py -d FS -ac all -c ${params[$i]} -ne 20 -g 0 --seed 0 --group sens_K --rho 0.2 -ut 0.09 -et 0.06 -ee 0.01 -at 0.3 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.04 --wandb
	python3 train.py -d desktop_assembly -ac all -c ${params[$i]} -ne 30 -g 1 --seed 0 --group sens_K --rho 0.3 -ut 0.09 -at 0.3 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 --wandb -ls 512 128 40
done

