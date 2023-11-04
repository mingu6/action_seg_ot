#!/bin/bash

actions=("changing_tire" "coffee" "cpr" "jump_car" "repot")
clusters=(11 10 7 12 8)
seed=0
gpu=0

for i in ${!actions[@]}; do
	python3 train_ours.py -d YTI -ac ${actions[$i]} -c ${clusters[$i]} -ps 5 -ne 10 -g $gpu --seed $seed --group paper --rho 0.1 -uw 0.05 -bn -et 0.06 -ee 0.01 -a 0.3 -ht 0. -vf 5 -lr 1e-3 -wd 1e-4 -pf 1.00 --ema 0.  -lc -r 10 --wandb -ls 3000 32 32 -x -1
done
