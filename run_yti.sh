#!/bin/bash

actions=("changing_tire" "coffee" "cpr" "jump_car" "repot")
clusters=(11 10 7 12 8)
seed=0
gpu=0

for i in ${!actions[@]}; do
	python3 train_ours.py -d YTI -ac ${actions[$i]} -c ${clusters[$i]} -ps 10 -ne 20 -g $gpu --seed $seed --group setting_13 --rho 0.1 -uw 0.05 -bn -ps 10 -et 0.06 -ee 0.02 -a 0.1 -ht 0.2 -vf 1 -lr 1e-3 -wd 1e-4 -pf 0.25 -lc -r 15 --wandb -ls 3000 32 32 -x -1
done
