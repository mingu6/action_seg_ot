#!/bin/bash

actions=("pancake" "salat" "friedegg" "scrambledegg" "sandwich" "juice" "milk" "tea" "cereals" "coffee")
clusters=(14 8 9 12 9 8 5 7 5 7)
seed=0
gpu=0

for i in ${!actions[@]}; do
	# python3 train.py -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 15 -g $gpu --seed 0 -s --rho 0.2 -lat 0.09 -r 0.04 -lr 1e-3 -wd 1e-4 -vf 5 --group improved_solver --wandb -v -ua
	python3 train.py -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 15 -g $gpu --seed 0 -s --rho 0.2 -lat 0.1 -r 0.04 -ae 0.7 -at 0.4 -lr 1e-3 -wd 1e-4 -vf 5 --group improved_solver --wandb -v -ua
	# python3 train.py -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 15 -g $gpu --seed 0 -s --rho 0.15 -ut 0.05 -at 0.4 -ae 0.6 -r 0.04 -lr 1e-3 -wd 1e-4 -vf 5 --group improved_solver --wandb -v
done
