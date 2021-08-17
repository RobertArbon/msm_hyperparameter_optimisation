#!/bin/bash


conda activate msmsense
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


python create_paths.py 1fme > 1fme.log > 2>&1 & 
python create_paths.py 2f4k > 2f4k.log > 2>&1 & 
python create_paths.py 2jof > 2jof.log > 2>&1 & 
python create_paths.py 2wav > 2wav.log > 2>&1 &
