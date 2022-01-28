#!/bin/bash

#  -h, --help            show this help message and exit
#  -i SEARCH_SPACE, --search-space SEARCH_SPACE
#                        Path to file that defines the HP search space dictionary
#  -n NUM_TRIALS, --num-trials NUM_TRIALS
#                        Number of HP trials
#  -o OUTPUT_FILE, --output-file OUTPUT_FILE
#                        Path to hd5 file to store HP samples
#  -s SEED, --seed SEED  Random seed

source ~/.bashrc
conda activate msmsense

msmsense hyperparameters \
 -i searchspace.yaml \
 -n 100 \
 -o hpsample.h5 \
 -s 278346826
