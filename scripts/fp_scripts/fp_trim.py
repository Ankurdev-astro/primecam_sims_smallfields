# Imports
import numpy as np
import os
import argparse
from filelock import FileLock
from astropy.table import QTable

def read_full_table(file_path):
    try:
        return QTable.read(file_path, path='dettable_stack')
    except Exception as e:
        print(f"Error reading the table: {e}")
        exit(1)

def select_wafer(dettable, wafer_slot):
    return dettable[dettable['wafer_slot'] == wafer_slot]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Select detectors from the w2 wafer.")
    parser.add_argument(
        "ndets_selected", 
        type=int, 
        help="Number of detectors to select. Must be an even number and >= 100."
    )
    args = parser.parse_args()
    if args.ndets_selected < 100 or args.ndets_selected % 2 != 0:
        raise argparse.ArgumentTypeError("ndets_selected must be an even number and at least 100.")
    return parser.parse_args()

def validate_selection(ndets_selected, total_dets):
    if ndets_selected > total_dets:
        raise ValueError(f"Error: ndets_selected ({ndets_selected}) cannot exceed "
                         f"the total number of detectors in w2 ({total_dets}).")

def generate_trimmed_table(trim_dettable_w2, ndets_selected):
    first_index = trim_dettable_w2["index"][0]
    last_index = trim_dettable_w2["index"][-1]
    pairs_select = ndets_selected // 2
    linspace_indices = np.linspace(first_index, last_index, pairs_select, dtype=int)
    pixels_select = trim_dettable_w2[np.isin(trim_dettable_w2["index"], linspace_indices)]['pixel']
    pixel_strings = [f"{int(pixel):04}".encode('utf-8') for pixel in pixels_select]
    mask = np.isin(trim_dettable_w2['pixel'], pixel_strings)
    return trim_dettable_w2[mask]

def write_trimmed_table(sel_dettable, file_path):
    sel_dettable.write(file_path, path='dettable_trim', serialize_meta=True, overwrite=True)

def main():
    fp_dir = "./input_files/fp_files/"
    # print(f"FP directory: {fp_dir}")
    hf_fulltable_file = os.path.join(fp_dir, "fp_f280_dettable.h5")
    # print(f"Full detector table file: {hf_fulltable_file}")
    dettable_full = read_full_table(hf_fulltable_file)
    trim_dettable_w2 = select_wafer(dettable_full, 'w2')
    # print(f"Number of detectors in w2: {len(trim_dettable_w2)}")

    args = parse_arguments()
    validate_selection(args.ndets_selected, len(trim_dettable_w2))

    # print(f"Selecting {args.ndets_selected} detectors from w2.")
    dets_trim_filename = f"dets_FP_PC280_{args.ndets_selected}_w2.h5"
    
    hf_trimtable_file = os.path.join(fp_dir, dets_trim_filename)
    # lockfile = hf_trimtable_file + ".lock"
    # lock = FileLock(lockfile)

    # with lock:
    #     if not os.path.exists(hf_trimtable_file):
    #         # print("Generating trimmed detector table file ...")
    #         sel_dettable = generate_trimmed_table(trim_dettable_w2, args.ndets_selected)
    #         print(f"Number of selected detectors: {len(sel_dettable)}")
    #         print(f"Writing trimmed detector table to {hf_trimtable_file} ...")
    #         write_trimmed_table(sel_dettable, hf_trimtable_file)
        
    # if os.path.exists(lockfile):
    #     os.remove(lockfile)
    

    if not os.path.exists(hf_trimtable_file):
        # print("Generating trimmed detector table file ...")
        sel_dettable = generate_trimmed_table(trim_dettable_w2, args.ndets_selected)
        print(f"Number of selected detectors: {len(sel_dettable)}")
        print(f"Writing trimmed detector table to {hf_trimtable_file} ...")
        write_trimmed_table(sel_dettable, hf_trimtable_file)

if __name__ == "__main__":
    main()
