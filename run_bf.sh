#!/bin/bash

actions=("pancake" "salat" "friedegg" "scrambledegg" "sandwich" "juice" "milk" "tea" "cereals" "coffee")
clusters=(14 8 9 12 9 8 5 7 5 7)
seed=0
gpu=0

for i in ${!actions[@]}; do
	python3 train_ours.py -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ps 10 -ne 15 -g $gpu --seed 0 --group debug4 --rho 0.1 -uw 0.05 -bn -et 0.06 -ee 0.01 -a 0.3 -ht 0. -vf 5 -lr 1e-3 -wd 1e-4 -pf 1. -lc -r 10 --wandb -s --ema 0.
done
