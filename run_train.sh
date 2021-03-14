#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --dataset multi --target $4 --net $3 --save_check
