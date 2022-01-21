#!/bin/bash

export protein=$1
export first=$2
export last=$3

conda activate msmsense
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for i in  $(seq $first $last) 
do 
	python score.py $protein $i > $protein.log 2>&1
done
