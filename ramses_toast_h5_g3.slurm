#!/bin/bash

###############################################
###             SLURM script for           ####
###             TOAST H5 to G3             ####
###                                        ####
###    Based on Ramses FB mapmaking setup  ####
###############################################

### This script converts a TOAST HDF5 simulation directory to G3.

### section 1 - SLURM params

#SBATCH --partition=smp
#SBATCH --account=ag-riechers # For Ramses/ CCAT

#SBATCH --ntasks-per-node 4 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 2 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes

#SBATCH --job-name=toast_h5_g3 # Job name

#SBATCH --mem=100G           # max mem requested, per node
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-00:30:00

#SBATCH --output ./logs/%j.%x.out   # stdout file (overwrite)
#SBATCH --error ./logs/%j.%x.err     # stderr file (overwrite)
### To handle dir ownership issues
umask 0002
mkdir -p logs
chmod g+rws logs

### section 2 - Runtime env set-up
echo ""
echo "*************"
echo "Loading Modules and Env..."
module purge
### load all modules needed for Job
module load lang/Miniconda3/23.9.0-0
module load lib/mpi4py/3.1.5-gompi-2023b
### initiate conda
conda deactivate
conda activate toast3
echo "*************"

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
echo "mpi4py import version: $(python -c 'import mpi4py; print(mpi4py.__version__)')"
echo "Using GCC lib: $(which gcc)"
echo ""

### section 4 - Set Job parameters
H5_DIR="./ccat_datacenter_mock/mockdata/orionA_ATMdata_d100/sim_PCAM280_h5_ORIONA_2026_07_15_d100/"

### section 5 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "Converting TOAST H5 to G3..."
echo ""
python scripts/helper_scripts/toast_h5_g3.py "$H5_DIR"

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"

