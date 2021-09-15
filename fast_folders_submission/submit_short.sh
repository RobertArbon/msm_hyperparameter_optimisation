#!/bin/bash
#BSUB -W 24:00
#BSUB -R "rusage[mem=4]"
#BSUB -n 34 
#BSUB -R "span[ptile=34]"
#BSUB -q cpuqueue

source ~/.bashrc
conda activate msmsense 

export PYEMMA_NJOBS=1
export OMP_NUM_THREADS=1

hps="hpsample.h5"
basedir="/data/chodera/arbon/fast_folders"
thisdir=`pwd`
topfile=DESRES-Trajectory_${job^^}-0-protein_xtc_1ns/${job^^}-0-protein/protein.pdb
trajglob=DESRES-Trajectory_${job^^}-*-protein_xtc_1ns/${job^^}-*-protein/*.xtc
outdir="${basedir}/${job}"

#test -f ${basedir}/${topfile} && echo "Top file found: ${basedir}/${topfile}" 
#test -f ${hps} & echo "HP sample file found: ${hps}" 
#echo $(ls -l "${basedir}/${trajglob}" |  wc -l) "trajectory files found" 
#echo "Output directory" ${basedir}/${job} 

time  msmsense count_matrices -i ${hps} -d ${basedir} -t ${topfile} -g ${trajglob} -r 102 -n 34 -l 10:101:10 -o ${outdir} -s 928473927

