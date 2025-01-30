import argparse
from mpi4py import MPI
import os
from scripts.helper_scripts.ccat_logger import get_ccat_logger

from sotodlib import core, coords
import numpy as np
from pixell import enmap

# Map parameters
DEG = np.pi / 180.0
wcsk = coords.get_wcs_kernel('car', 16, -2.0, 0.5 / 60.0 * DEG)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = get_ccat_logger(rank)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--context',
        default='ccat_datacenter_mock/context'
    )
    args = parser.parse_args()
    
    # from so3g import proj

    ct_yaml = os.path.join(args.context, 'context.yaml')
    
    if rank == 0:
        logger.ml_pipeline('Init context...')
    ct = core.Context(ct_yaml)

    comm.barrier()
    
    if rank == 0:
        logger.ml_pipeline('Load obs list...')
    obs = ct.obsdb.query()
    comm.barrier()

    # Broadcast the obs list to all ranks
    obs = comm.bcast(obs, root=0)

    # Distribute observations among ranks
    geoms = []
    for idx_o, o in enumerate(obs):
        if idx_o % size == rank:
            logger.ml_pipeline(f"Processing Obs #{idx_o} ...")
            #print(f"\n \t Observation details: {o}")
            tod = ct.get_obs(o)
            # Promote TOD?
            fm = core.FlagManager.for_tod(tod)
            P = coords.P.for_tod(tod)
            geom = coords.get_footprint(tod, wcsk)
            geoms.append(geom)

    # Gather all geoms to rank 0
    all_geoms = comm.gather(geoms, root=0)

    if rank == 0:
        # Flatten the list of geoms
        flat_geoms = [geom for sublist in all_geoms for geom in sublist]
        supergeom = coords.get_supergeom(*flat_geoms)
        logger.ml_pipeline(f'\n Computed Geometry... {supergeom}')

        # print(supergeom) 
        map_file = os.path.join(args.context, "geom.fits")
        logger.ml_pipeline(f"Writing map geometry to {map_file}")
        enmap.write_map_geometry(map_file, *supergeom)
        logger.ml_pipeline("Map geometry written successfully.")

if __name__ == "__main__":
    main()
