#!/bin/bash

actions=("pancake" "salat" "friedegg" "scrambledegg" "sandwich" "juice" "milk" "tea" "cereals" "coffee")
clusters=(14 8 9 12 9 8 5 7 5 7)
seed=0
gpu=0

for i in ${!actions[@]}; do
	python3 src/train.py -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 15 -g $gpu --seed 0 -s --rho 0.2 -lat 0.1 -r 0.04 -ae 0.7 -at 0.4 -lr 1e-3 -wd 1e-4 -vf 5 --group main_results --wandb -v -ua
done
