# primecam_sims_smallfields
**If you intend to use these simulations, please get in touch with the corresponding author for further information or collaboration:  
Ankur Dev (adev@uni-bonn.de)**
## Simulations for PrimeCam small fields: Timestream data and Mapmaking

Note: High memory usage expected. Run only on a cluster

Auxiliary files needed: 
[pysm3_map_nside2048_allStokes.fits](https://www.dropbox.com/scl/fi/gm4xuhguht5dx848d9e69/pysm3_map_nside2048_allStokes.fits?rlkey=0qga1dkj6442vxrnvku3pcrlx&dl=0)

This file should be put inside `./input_files/`

```
curl -L -o input_files/pysm3_map_nside2048_allStokes.fits "<file_url>"
```
Else, input your own sky maps in the format required by TOAST

### First let's start with timestream simulation.
*Much of the documentation here is work in progress*

As an example, our first target field case is **OrionA**. We can choose to either scan the field with Constant Elevation Scans(CES), or choose to vary elevation and have a Lissajous-like scan pattern. Pong and Daisy (SCUBA-2 like patterns) have not been implemented yet in these set of simulations.
##### Usage: 

```
export OMP_NUM_THREADS=NUM_THREADS

mpirun -np N_PROCS python sim_data_primecam_mpi.py --sch SCHEDULE_FILE --dets NUM_DETS
```

NUM_DETS is the number of detectors to be used in the simulation. Currently supported values range from 100 to 500, in increments of 100.
The GROUP_SIZE for the problem is calculated based on the runtime parameters. User may set group size as:

```
mpirun -np N_PROCS python sim_data_primecam_mpi.py -s SCHEDULE_FILE -d NUM_DETS -g GROUP_SIZE
```

Note: Provide only the schedule file name, not the full path.

For help:
```
python sim_data_primecam_mpi.py -h
```

This shall create 'ccat_datacenter_mock' dir. All observations are stored here. 
Context dir and Maximum-Likelihood Map outputs will be stored here.

All simulated data will be saved in `./ccat_datacenter_mock/data_testmpi/deep56_data_d{NUM_DETS}`, with the required directories created automatically by the code.

An example SLURM script is provided that uses ARRAY JOBS to process all schedules listed in `./input_files/schedules`. 

### Next: Processing and Map-making pipeline:

##### Usage:
*Much of the documentation here is work in progress. A filter and bin map-maker will be implemented here.*


### Converting TOAST HDF5 files to SPT3G format:

TOAST natively supports writing data to disk in HDF5 files.
To convert the produced h5 files into g3, use `scripts/toast_h5_g3.py`

```
Usage: python toast_h5_g3.py [-h] h5_dirs
Example: python toast_h5_g3.py ../ccat_datacenter_mock/path_to_h5_dir
```

This script takes a directory containing h5 files and generates a directory with
corresponding g3 files. If there are multiple h5 directories that need to be processed,
run this script for each of those. The script shall throw an error if the corresponding 
g3 file already exists.

