#!/bin/bash

#submit=submit_short.sh
#for pdb in cln025 
#do
#  export job=${pdb}
#  bsub -J ${job} -e %J.${job}.out -o %J.${job}.out < ${submit}
#done

submit=submit_long.sh
for pdb in 2f4k 2wav 2jof 
do
  export job=${pdb}
  bsub -J ${job} -e %J.${job}.out -o %J.${job}.out < ${submit}
done
