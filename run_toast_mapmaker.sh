#!/bin/bash
export OMP_NUM_THREADS=2

echo "$(which python)"
echo "$(python --version)"

###################
####  CONFIG   ####
###################

### for arc10 tests
INDIR='orionA_ATMdata_d100'
OUTDIR='fb_dump1'
GRP_SIZE=24
NOTES=''
###################

mpirun -np 48 python write_toast_maps.py -in $INDIR -out $OUTDIR -g $GRP_SIZE -n "$NOTES"