#!/bin/bash
### This is a comment in a SLURM script
# #SBATCH is a directive for the SLURM scheduler.

### section 1 - SLURM params

###SBATCH --partition=aifa-science  # for the main queue
### or
#SBATCH --partition=science-old  # alternate old

#SBATCH --ntasks 8        # number of procs to start(Total)
#SBATCH --cpus-per-task 2 # number of cores per task(Threads per task)
#SBATCH --nodes 1         # number of nodes

### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task
### #ntasks = #nodes x #tasks-per-node , must be met
### #SBATCH --ntasks-per-node 4 # max tasks per node

#SBATCH --mem-per-cpu=2G # mem per proc
### #SBATCH --mem=100G # max mem requested
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-01:00:00 #estimated runtime max

#SBATCH --job-name mpi_toast_test  # Job name
### #SBATCH --mail-type=ALL   # notifications for job done & fail
### #SBATCH --mail-user=adev@astro.uni-bonn.de #user email for updates
#SBATCH -o ./logs/%j.out # STDOUT                                             
#SBATCH -e ./logs/%j.err # STDERR 

### If you have a specific Python environment, activate it here
### source activate myenv

module purge
module load gcc openmpi slurm
# Or for Anaconda
# Intialize Conda, need the Conda shell file loc
source /vol/aibn49/data1/adev/opt/anaconda3/etc/profile.d/conda.sh
# Activate your environment
conda activate toast3-alone

# Set environment variables

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


# In case you are using libraries from a different location specified
# in your library path, then you need to add them here.
# export LD_LIBRARY_PATH=<...>:$LD_LIBRARY_PATH

echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""

# Run python code, the program with flags
###export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
###python sim_data_primecam_mpi.py --test-run
mpirun -np $SLURM_NTASKS python sim_data_primecam_mpi.py --test-run

# If Without mpi
# python3 matplotlib_test.py

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""

exit 0

