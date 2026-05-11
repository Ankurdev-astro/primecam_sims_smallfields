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

PARENT_DIR="ccat_datacenter_mock"
DATA_DIR="mockdata_test1"
MAP_DIR="fb_maps"
FULL_DATA_PATH="${PARENT_DIR}/${DATA_DIR}"
FULL_MAP_PATH="${PARENT_DIR}/${MAP_DIR}"

FULL_INDIR="${FULL_DATA_PATH}/${INDIR}"
FULL_OUTDIR="${FULL_MAP_PATH}/${OUTDIR}"
GRP_SIZE=24
NOTES=''
###################

echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

### Build Map
echo ""
echo "Processing Filter and Bin Pipeline..."
echo ""
mpirun -np 48 python write_toast_maps.py -in $FULL_INDIR -out $FULL_OUTDIR -g $GRP_SIZE -n "$NOTES"

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"
