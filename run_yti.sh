#!/bin/bash

actions=("changing_tire" "coffee" "cpr" "jump_car" "repot")
clusters=(11 10 7 12 8)
seed=0
gpu=1

for i in ${!actions[@]}; do
	python3 train.py -d YTI -ac ${actions[$i]} -c ${clusters[$i]} -ne 10 -g $gpu --seed $seed --rho 0.15 -lat 0.12 -r 0.04 -ua -lr 1e-3 -wd 1e-4 -ls 3000 32 32 -x -1 -vf 5  --group main_results --wandb -v
done
