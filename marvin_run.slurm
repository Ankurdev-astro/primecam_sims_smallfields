#!/bin/bash

###############################################
### Example SLURM script for primecam_sims ####
###                                        ####
###     Tested on Uni Bonn Marvin Cluster  ####
###             v1.1 , 16.10.2024          ####
###############################################

### section 1 - SLURM params

#SBATCH --partition=intelsr_short  # cluster specific / intelsr_short, lm_short
### https://wiki.hpc.uni-bonn.de/en/running_jobs

### #SBATCH --ntasks 8      # number of procs to start
### or
#SBATCH --ntasks-per-node 8 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 12 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes

#SBATCH --job-name=n8c12_s9det100 # Job name, #ntaskstotal#corespertask
### ntasks = 8, cpus=12, s9det100 seems best ~ 60min

### #ntasks = #nodes x #ntasks-per-node
### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task

#SBATCH --mem-per-cpu=6G # mem per proc
### #SBATCH --mem=130G                # max mem requested, per node
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-01:30:00 #estimated runtime max

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
conda activate toast3

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

### section 4 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""
###python3 matplotlib_test.py
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun -np $SLURM_NTASKS python3 sim_data_primecam_mpi.py -s schedule_1scan.txt


echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"

# Log job resource usage into the output file
echo ""
echo "***********************"
echo "****** RES USAGE ******"

