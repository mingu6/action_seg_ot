#!/bin/bash

actions=("pancake" "salat" "friedegg" "scrambledegg" "sandwich" "juice" "milk" "tea" "cereals" "coffee")
clusters=(14 8 9 12 9 8 5 7 5 7)
seed=0
gpu=0

for i in ${!actions[@]}; do
	# python3 train_ours.py -d ${actions[$i]} -c ${clusters[$i]} -lc -ps 10 -ne 15 -g $gpu --seed $seed --wandb --group tuned --rho 0.05 -uw 0.1 -ee 0.04 -pf 0.5
	# python3 train_ours.py -d pancake -c 14 -ps 10 -ne 15 -g 0 --seed 0 --group main1 --rho 0.05 -uw 0.1 -ee 0.04 -a 0.5 -vf 1 -lr 1e-3 -wd 1e-4 --wandb -lc -pf 0.5
	python3 train_ours.py -d ${actions[$i]} -c ${clusters[$i]} -ps 10 -ne 20 -g $gpu --seed $seed --group setting_22 --rho 0.1 -uw 0.03 -bn -ps 10 -et 0.04 -ee 0.02 -a 0.15 -ht 0.1 -vf 5 -lr 1e-3 -wd 1e-4 -pf 0.25 -lc -r 15 --wandb -nt 10 3
	 --wandb
done

