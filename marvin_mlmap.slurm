#!/bin/bash

###############################################
###       SLURM script for ML Mapmaking    ####
###                                        ####
###     Tested on Uni Bonn Marvin Cluster  ####
###             v1.0 , 18.10.2024          ####
###############################################

### section 1 - SLURM params

#SBATCH --partition=lm_short  # cluster specific / intelsr_short, lm_short
### https://wiki.hpc.uni-bonn.de/en/running_jobs

### #SBATCH --ntasks 8      # number of procs to start
### or
#SBATCH --ntasks-per-node 16 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 6 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes

#SBATCH --job-name=det200_n16c6_ml # Job name, #ntaskstotal#corespertask
### ntasks = 8, cpus=12, s9det100 seems best ~ 60min

### #ntasks = #nodes x #ntasks-per-node
### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task

#SBATCH --mem-per-cpu=10G # mem per proc
### #SBATCH --mem=130G                # max mem requested, per node
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-07:30:00 #estimated runtime max

### #SBATCH --mail-type=ALL   # notifications for job done & fail
### #SBATCH --mail-user=adev@astro.uni-bonn.de #user email for updates
#SBATCH --output ./logs/%j.%x.out   # stdout file (overwrite)
#SBATCH --error ./logs/%j.%x.err     # stdout file (overwrite)


### section 2 - Runtime env set-up
module purge
### load all modules needed for Job
module load mpi4py
### initiate conda
### source /opt/software/easybuild-INTEL/software/Anaconda3/2022.05/etc/profile.d/conda.shi
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda deactivate
conda activate mlmap

### section 3 - Job Logging
echo ""
echo "*************"
echo "Running Job..."
echo "Starting at `date`"
echo "Hostname $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Slurm Ntasks: $SLURM_NTASKS"
echo "Number of Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Number of CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Cores per Node: $SLURM_CPUS_ON_NODE"
echo "Total Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Current working directory is `pwd`"
echo "Python path: $(which python)"
echo "Using MPI lib: $(which mpirun)"
echo "Using GCC lib: $(which gcc)"
echo ""

PARENT_DIR="arc10_data_d200"
DATA_DIR="ccat_datacenter_mock/data_testmpi/"
FULL_DATA_PATH="${DATA_DIR}${PARENT_DIR}"
ML_CONFIG="config.yaml"

### section 4 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### Build context data
echo ""
echo "Building context data ..."
echo ""
mpirun -np $SLURM_NTASKS python write_context_primecam_mpi.py $FULL_DATA_PATH

### Build Map Footprint
echo ""
echo "Building map footprint ..."
echo ""
mpirun -np $SLURM_NTASKS python write_footprint_primecam_mpi.py

### Build Map
echo ""
echo "Building ML map ..."
echo ""
mpirun -np $SLURM_NTASKS python make_ml_map_primecam.py --config $ML_CONFIG

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"

