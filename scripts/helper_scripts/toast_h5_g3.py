#Author: Ankur Dev
#Last Updated: April 26, 2024
#Adapted from: https://github.com/hpc4cmb/toast/blob/toast3/src/toast/scripts/toast_hdf5_to_spt3g.py

"""
This script loads HDF5 format directories and exports to SPT3G format data.
"""

import toast
import toast.io as io
import toast.ops

import re
import os
import argparse
import numpy as np
from toast.observation import default_values as defaults
from spt3g import core as c3g
from toast import spt3g as t3g

from g3_helper import export_obs_meta_mod


def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Convert TOAST h5 files from primecam_sims to g3"
            )
    parser.add_argument('h5_dirs', type=str, help = "Path to TOAST HDF5 dir")
    args = parser.parse_args()
    return(args)


def main():
    log = toast.utils.Logger.get()

    timer = toast.timing.Timer()
    timer.start()

    toast_comm = toast.Comm(world=comm, groupsize=1)
    # Shortcut for the world communicator
    world_comm = toast_comm.comm_world

    # Create the (empty) data
    data = toast.Data(comm=toast_comm)

    # Parse arguments
    args = parse_arguments()
    h5_dirs_path = os.path.abspath(args.h5_dirs)

    #count = 0
    # Load the h5 data.
    log.info_rank(
        f"Loading HDF5 data from {h5_dirs_path}", comm=world_comm
    )
    
    for h5files in os.listdir(h5_dirs_path):
        # Load h5 file and make path
        h5_file_path = os.path.join(h5_dirs_path, h5files)
        if os.path.isfile(h5_file_path) and h5files.endswith(".h5"):
            #count += 1
            # Load the observation from the HDF5 file
            log.info_rank(
                f"Added h5 file: {os.path.relpath(h5_file_path)}", 
                comm=world_comm)

            obs = io.load_hdf5(path=h5_file_path, comm=toast_comm)
            # Append the observation
            data.obs.append(obs)
            #if count >= 5:
            #    break

    # Build up the lists of objects to export from the first observation

    noise_models = list()
    meta_arrays = list()
    shared = list()
    detdata = list()
    intervals = list()

    msg = "Exporting observation fields:"

    ob = data.obs[0]
    for k, v in ob.shared.items():
        g3name = f"shared_{k}"
        if re.match(r".*boresight.*", k) is not None:
            # These are quaternions
            msg += f"\n  (shared):    {k} (quaternions)"
            shared.append((k, g3name, c3g.G3VectorQuat))
        elif k == defaults.times:
            # Timestamps are handled separately
            continue
        else:
            msg += f"\n  (shared):    {k}"
            shared.append((k, g3name, None))

    for k, v in ob.detdata.items():
        msg += f"\n  (detdata):   {k}"
        detdata.append((k, k, None))

    for k, v in ob.intervals.items():
        if k != "ALL_OBSERVATION_SAMPLES":
            msg += f"\n  (intervals): {k}"
            intervals.append((k, k))

    for k, v in ob.items():
        if isinstance(v, toast.noise.Noise):
            msg += f"\n  (noise):     {k}"
            noise_models.append((k, k))
        elif isinstance(v, np.ndarray) and len(v.shape) > 0:
            if isinstance(v, u.Quantity):
                raise NotImplementedError("Writing array quantities not yet supported")
            msg += f"\n  (meta arr):  {k}"
            meta_arrays.append((k, k))
        else:
            msg += f"\n  (meta):      {k}"
    
    log.info_rank(msg, comm=world_comm)
    
    # Export the data
    parent_dir = os.path.dirname(h5_dirs_path)
    basedir = os.path.basename(h5_dirs_path)

    save_spt3g_dir = os.path.join(parent_dir, basedir.replace('h5', 'g3'))
    try:
        os.makedirs(save_spt3g_dir, exist_ok=False)
    except Exception as e:
        #print(f"Error: {e}")
        raise RuntimeError("Error: g3 directory already exists!")

    meta_exporter = export_obs_meta_mod(
     noise_models=noise_models,
        meta_arrays=meta_arrays,
    )

    data_exporter = t3g.export_obs_data(
        shared_names=shared,
        det_names=detdata,
        interval_names=intervals,
        compress=True,
    )
    exporter = t3g.export_obs(
        meta_export=meta_exporter,
        data_export=data_exporter,
        export_rank=0,
    )

    g3_dumper = toast.ops.SaveSpt3g(
    directory=save_spt3g_dir, obs_export=exporter
    )
    g3_dumper.purge = True
    g3_dumper.gzip = False

    log.info_rank(
        f"Exporting SPT3G data to {os.path.relpath(save_spt3g_dir)}", comm=world_comm)
    g3_dumper.apply(data)
   
    log.info_rank(
            "Wrote g3 files in", comm=world_comm, timer=timer)

if __name__ == "__main__":
    comm, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=comm):
        main()
