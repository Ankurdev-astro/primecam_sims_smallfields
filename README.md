# PrimeCam Small-Field Simulations

**If you intend to use these simulations, please get in touch with the corresponding author for further information or collaboration:  
Ankur Dev (ankurdev@cornell.edu)**

This repository contains a TOAST3-based workflow for PrimeCam / FYST small-field
simulations. The current pipeline generates mock detector timestreams from
schedule files and then runs a filter-and-bin mapmaker over the resulting TOAST
HDF5 observations.

The workflow is intended for cluster use. Memory usage can be high, especially
when simulating many detectors and atmospheric timestreams.

## Main Workflow

1. Prepare the input sky map, schedule files, and focal-plane files.
2. Run `ramses_mockdata_sim.slurm` to simulate timestream data for each schedule.
3. Run `ramses_fbmap.slurm` to load the HDF5 observations, filter the TOD, and
   write map products.
4. Optionally convert TOAST HDF5 observations to SPT3G `.g3` files.

The main scripts are:

- `sim_data_primecam_mpi.py`: MPI timestream simulation driver.
- `write_toast_maps.py`: TOAST filter-and-bin mapmaker.
- `ramses_mockdata_sim.slurm`: Ramses SLURM array job for simulations.
- `ramses_fbmap.slurm`: Ramses SLURM job for mapmaking.
- `scripts/fp_scripts/fp_prepare.py`: returns prepared detector-subset QTables in memory.
- `scripts/helper_scripts/toast_h5_g3.py`: converts TOAST HDF5 observations to
  SPT3G format.

## Inputs

### Sky Map

The default simulation scans a PySM3 HEALPix full-Stokes map:

[`pysm3_map_nside2048_allStokes.fits`](https://www.dropbox.com/scl/fi/gm4xuhguht5dx848d9e69/pysm3_map_nside2048_allStokes.fits?rlkey=0qga1dkj6442vxrnvku3pcrlx&dl=0)

Place it in:

```bash
input_files/pysm3_map_nside2048_allStokes.fits
```

For example:

```bash
mkdir -p input_files
curl -L -o input_files/pysm3_map_nside2048_allStokes.fits "<file_url>"
```

You may also provide your own input sky map, but it must be in the format
expected by TOAST's `ScanHealpixMap` operator.

### Schedules

Schedule files live in:

```text
input_files/schedules/
```

The current OrionA example is:

```text
input_files/schedules/schedule_ORIONA_2026_07_15.txt
```

As an example, the first target field case is **OrionA**. The schedules can scan
the field with constant-elevation scans (CES) or use varying elevation for a
Lissajous-like pattern. Pong and Daisy / SCUBA-2-like patterns have not yet been
implemented in this simulation set.

When calling `sim_data_primecam_mpi.py` directly, pass only the schedule file
name, not the full path.

### Focal Plane

The module name is required. It selects a detector table bundled with
`primecam-focalplane`; detectors are then selected from wafer `w2`. No local
detector file is needed.

- Set `--module-name <MODULE_NAME>` (for example, `primecam_f280`).
- Set `-d <NUM_DETS>` to an even number from 10 up to the wafer's detector count.
- Add `--tube-offset` to apply the module-to-boresight offset, or set
  `TUBE_OFFSET=TRUE` for Slurm runs.
- For a custom table, pass `file_path` to `prepare_trimmed_table()` or use
  `--file-path` when running `scripts/fp_scripts/fp_prepare.py` directly.

## Timestream Simulation

The simulation driver is:

```bash
sim_data_primecam_mpi.py
```
Several parameters can be set within `class Args`

Basic direct usage:

```bash
export OMP_NUM_THREADS=<NUM_THREADS>
mpirun -np <N_PROCS> python sim_data_primecam_mpi.py \
  --sch <SCHEDULE_FILE> \
  --dets <NUM_DETS> \
  --module-name <MODULE_NAME>
```

For help:

```bash
python sim_data_primecam_mpi.py -h
```

The code creates the data-center mock directory automatically. Current output
paths are of the form:

```text
<PARENT_DIR>/<DATA_DIR>/<RUN_NAME>_d<NUM_DETS>_<MODULE_NAME>/<SIM_OUTPUT_DIR>/
```

Each observation is written as TOAST HDF5 with detector `signal` and `flags`
datasets.

## Running Simulations on Ramses

Use:

```bash
sbatch ramses_mockdata_sim.slurm
```

Before submitting, check Section 4 of `ramses_mockdata_sim.slurm`:

```bash
ndets=100
MODULE_NAME=primecam_f280
```

The script:

- Loads the Ramses `Miniconda3` and `mpi4py` modules.
- Activates the `toast3` conda environment.
- Creates `logs_sims/`.
- Collects all `*.txt` schedule files from `input_files/schedules/`.
- Uses `SLURM_ARRAY_TASK_ID` to select a schedule.
- Generates the trimmed focal-plane file for `ndets`.
- Runs `sim_data_primecam_mpi.py` with `mpirun`.

The current array setting is:

```bash
#SBATCH --array=0
```

Increase this, for example to `0-10`, when running multiple schedules.

## Filter-and-Bin Mapmaking

The mapmaking driver is:

```bash
write_toast_maps.py
```

It loads TOAST HDF5 observations, applies filtering, and bins to CAR maps.
Current filtering steps are:

- Common-mode removal with `toast.ops.CommonModeFilter`.
- 2D polynomial filtering with `toast.ops.PolyFilter2D(order=3)`.

Current binning uses:

- `toast.ops.PixelsWCS` with `projection="CAR"` and `auto_bounds=True`.
- Intensity-only mapmaking mode, `mode="I"`.
- `toast.ops.MapMaker`.

Map products currently written include:

- `orionA_hits.fits`
- `orionA_binmap.fits`
- `orionA_invcov.fits`
- `orionA_map.fits`
- `orionA_noiseweighted_map.fits`

Direct usage:

```bash
export OMP_NUM_THREADS=<NUM_THREADS>
mpirun -np <N_PROCS> python write_toast_maps.py \
  --input_dir <INPUT_DIR> \
  --output_dir <OUTPUT_DIR> \
  --grp_size <GROUP_SIZE> \
  --note_msg "<NOTES>"
```

For help:

```bash
python write_toast_maps.py -h
```

## Running Mapmaking on Ramses

Use:

```bash
sbatch ramses_fbmap.slurm
```

Before submitting, check Section 4 of `ramses_fbmap.slurm`:

```bash
INDIR='<INPUT_DIR_NAME>'
OUTDIR='<OUTPUT_DIR_NAME>'
PARENT_DIR='<PARENT_DIR>'
DATA_DIR='<DATA_DIR>'
GRP_SIZE=<GROUP_SIZE>
NOTES='<NOTES>'
```

With these settings, the input and output paths become:

```text
<PARENT_DIR>/<DATA_DIR>/<INPUT_DIR_NAME>
<PARENT_DIR>/<OUTPUT_DIR_NAME>
```

The script creates `logs_map/`, loads the same Ramses modules and `toast3`
environment, sets `OMP_NUM_THREADS`, and launches `write_toast_maps.py` through
`mpirun`.

## Converting TOAST HDF5 to SPT3G

TOAST natively writes observations to HDF5. To convert a produced HDF5
observation directory into SPT3G `.g3` files, use:

```bash
python scripts/helper_scripts/toast_h5_g3.py <H5_DIR>
```

Generic path example:

```bash
python scripts/helper_scripts/toast_h5_g3.py \
  <PARENT_DIR>/<DATA_DIR>/<INPUT_DIR_NAME>/<H5_OUTPUT_DIR>
```

The converter takes a directory containing `.h5` files and creates a sibling
directory whose name replaces `h5` with `g3`. If the destination G3 directory
already exists, the script raises an error rather than overwriting it. If there
are multiple HDF5 directories to process, run the converter once for each
directory.

## Notes

- Run these jobs on a cluster; local runs may exhaust memory quickly.
- Keep schedule files in `input_files/schedules/`.
- Keep focal-plane files in `input_files/fp_files/`.
- The SLURM scripts were tested on the Uni Koeln Ramses cluster.
- Much of this workflow is still evolving; check the script headers and SLURM
  Section 4 parameters before launching large production runs.
