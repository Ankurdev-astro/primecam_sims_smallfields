#!/bin/bash

# Exit immediately if a command exits with a non-zero status (error)
set -e

#********************#
# Set run parameters
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
export OMP_NUM_THREADS=8

nprocs=8
tod_data_dir="ccat_datacenter_mock/arc10_data_testmpi_488hz/"
ml_config="config.yaml"

#********************#

## Build context data
echo ""
echo "Building context data ..."
echo ""
mpirun -np $nprocs python write_context_primecam_mpi.py $tod_data_dir

## Build Map Footprint
echo ""
echo "Building map footprint ..."
echo ""
mpirun -np $nprocs python write_footprint_primecam_mpi.py 

## Build Map
echo ""
echo "Building ML map ..."
echo ""
mpirun -np $nprocs python make_ml_map_primecam.py --config $ml_config

## Run ML pipeline:
## /usr/bin/time -v ./run_mlmap_pipeline.sh 2>&1 | tee -a logs/mlmap_280924_arc10.log
