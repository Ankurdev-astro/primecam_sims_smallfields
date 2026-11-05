"""Prepare and trim detector QTables in memory without writing files."""
import numpy as np
import argparse
from astropy.table import QTable

def read_full_table(file_path):
    return QTable.read(file_path, path='dettable_stack')

def select_wafer(dettable, wafer_slot):
    return dettable[dettable['wafer_slot'] == wafer_slot]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Select detectors from a module's w2 wafer.")
    parser.add_argument(
        "ndets_selected", 
        type=int, 
        help="Number of detectors to select. Must be an even number and >= 10."
    )
    parser.add_argument(
        "--tube-offset",
        action="store_true",
        help="Apply the selected module's module-to-boresight offset before trimming."
    )
    parser.add_argument(
        "-m",
        "--module-name",
        required=True,
        help="PrimeCam module name from the primecam-focalplane configuration.",
    )
    parser.add_argument(
        "--file-path",
        help="Override the detector table bundled with primecam-focalplane."
    )
    args = parser.parse_args()
    return args

def validate_selection(ndets_selected, total_dets):
    if ndets_selected < 10 or ndets_selected % 2 != 0:
        raise ValueError("ndets_selected must be an even number and at least 10.")
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

def prepare_trimmed_table(
    ndets_selected,
    module_name,
    tube_offset=False,
    file_path=None,
):
    """Load, optionally offset, and trim the w2 wafer; return its QTable.

    By default, load the table bundled with primecam-focalplane. An explicit
    file_path overrides that source, with or without tube_offset enabled.
    module_name selects the configured module used to load the default table and
    to apply the optional module-to-boresight offset.
    """
    module = None
    if file_path is None or tube_offset:
        from primecam_focalplane.module_to_boresight import ModuleToBoresightOffset
        module = ModuleToBoresightOffset(module_name)

    dettable_full = module.load() if file_path is None else read_full_table(file_path)
    if tube_offset:
        dettable_full = module.apply(dettable_full)

    trim_dettable_w2 = select_wafer(dettable_full, 'w2')
    validate_selection(ndets_selected, len(trim_dettable_w2))
    return generate_trimmed_table(trim_dettable_w2, ndets_selected)


def main():
    args = parse_arguments()
    table = prepare_trimmed_table(
        args.ndets_selected,
        tube_offset=args.tube_offset,
        file_path=args.file_path,
        module_name=args.module_name,
    )
    print(f"Number of selected detectors: {len(table)}")
    return table


if __name__ == "__main__":
    main()
