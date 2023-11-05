#!/bin/bash

actions=("changing_tire" "coffee" "cpr" "jump_car" "repot")
clusters=(11 10 7 12 8)
seed=0
gpu=0

for i in ${!actions[@]}; do
	python3 train.py -d YTI -ac ${actions[$i]} -c ${clusters[$i]} -ne 10 -g $gpu --seed $seed --group main_results --rho 0.1 -ut 0.05 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.04 -ls 3000 32 32 -x -1 --wandb
done
