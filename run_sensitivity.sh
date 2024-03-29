#!/bin/bash

# params=(0.02 0.04 0.06 0.08 0.1) # gw radius
# params=(0.08 0.11 0.14 0.17 0.2) # unbalanced weight
# params=(0.1 0.3 0.5 0.7 0.9) # alpha
params=(0.1 0.15 0.2 0.25 0.3) # rho
# params=(0.03 0.05 0.07 0.09 0.11) # training eps
# params=(10 14 19 24 29) # number of classes

for i in ${!params[@]}; do
	# python3 train.py -d FSeval -ac all -c 12 -ne 30 -g 1 --seed 0 --group sens_eps --rho 0.15 -lat 0.11 -et ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	# python3 train.py -d FS -ac all -c 19 -ne 30 -g 1 --seed 0 --group sens_eps --rho 0.15 -lat 0.15 -et ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	# python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 -g 1 --seed 0 --group sens_eps --rho 0.25 -lat 0.16 -et ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ls 512 128 40 -ua -r 0.02
	# python3 train.py -d FSeval -ac all -c 12 -ne 30 -g 1 --seed 0 --group sens_lambda --rho 0.15 -lat ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	# python3 train.py -d FS -ac all -c 19 -ne 30 -g 1 --seed 0 --group sens_lambda --rho 0.15 -lat ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	# python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 -g 1 --seed 0 --group sens_lambda --rho 0.25 -lat ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ls 512 128 40 -ua -r 0.02
	python3 train.py -d FSeval -ac all -c 12 -ne 30 -g 1 --seed 0 --group sens_rho -lat 0.11 --rho ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	python3 train.py -d FS -ac all -c 19 -ne 30 -g 1 --seed 0 --group sens_rho -lat 0.15 --rho ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 -g 1 --seed 0 --group sens_rho -lat 0.16 --rho ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ls 512 128 40 -ua -r 0.02
	# python3 train.py -d FSeval -ac all -c 12 -ne 30 -g 1 --seed 0 --group sens_radius --rho 0.15 -lat 0.11 -r ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	# python3 train.py -d FS -ac all -c 19 -ne 30 -g 1 --seed 0 --group sens_radius --rho 0.15 -lat 0.15 -r ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ua
	# python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 -g 1 --seed 0 --group sens_radius --rho 0.25 -lat 0.16 -r ${params[$i]} -vf 5 -lr 1e-3 -wd 1e-4 --wandb -ls 512 128 40 -ua
done

