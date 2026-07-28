#!/usr/bin/env python3
"""
Alex Heindel
Date: 07/22/26
Sparse crossing-muon light-yield reconstruction example.

For each crossing-muon event, the code builds

    A[event, voxel] = path_length[event, voxel] * 2.1 MeV/cm,

and uses the detected-photon count as b[event]. The voxel light-yield map x is
then obtained from a non-negative least-squares fit. With --lambda-smooth > 0,
the fit also penalizes differences between neighboring voxels:

    min  0.5 ||A x - b||^2
       + 0.5 lambda sum_neighbors (x_j - x_k)^2,
    subject to x >= 0.

To be used with the .h5 file output by read_lardon_tracks.py. 
To use another dataset, update the file naming ... detector bounds, and any desired event cuts.

Requirements
------------
  * PyROOT
  * NumPy and SciPy
  * liang_barsky.C defining liang_barsky_3d

Example
-------
python3 pds_cal_gen_scipy_sparse_l2smooth.py \
    --data-dir data/angelo2_config \
    --ndiv 16 \
    --num-events 60000 \
    --lambda-smooth 1e6 \
    --outname solutions/angelo2_16x16x16_60000evts_lambda1e6.txt
"""

import argparse
import math
from array import array
from pathlib import Path

import numpy as np
import tables as tab
import ROOT
from scipy.optimize import lsq_linear 
from scipy.sparse import csr_matrix, vstack 


ROOT.gROOT.SetBatch(True)

# Dataset-specific configuration. These are the main values a user would
# replace when adapting the example to another detector or simulation sample.
# TRAJECTORY_TREE = "TrajCut/CRTTrajectoryCut"
# PHOTON_TREE = "XAresponse/DetectedPhotons"
DETECTOR_BOUNDS = {
    "x": (-400.0, 400.0),
    "y": (-425.0, 425.0),
    "z": (-300.0, 600.0),
}

# Angelo2 channels retained for the no-PMT reconstruction.
# SELECTED_OP_CHANNELS = {
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 22, 23,
# }

MIP_ENERGY_LOSS = 2.1  # MeV/cm
NO_INTERSECTION = -999.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and solve a sparse crossing-muon light-yield system with "
            "optional nearest-neighbor L2 smoothing."
        )
    )
    # parser.add_argument(
    #     "--data-dir",
    #     type=Path,
    #     default=Path("data/angelo2_config"),
    #     help="Directory containing the Angelo2 ROOT files.",
    # )
    # parser.add_argument(
    #     "--first-run",
    #     type=int,
    #     default=0,
    #     help="First Angelo2 run index.",
    # )
    # parser.add_argument(
    #     "--num-files",
    #     type=int,
    #     default=120,
    #     help="Number of sequential Angelo2 files to use.",
    # )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("out.h5"),
        help="HDF5 file containing the output from read_lardon_tracks.py.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=5000,
        help="Maximum number of trajectory entries to reconstruct.",
    )
    parser.add_argument(
        "--ndiv",
        type=int,
        default=16,
        help="Number of equal voxel divisions along each axis.",
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=0.0,
        help="Nearest-neighbor smoothness strength; zero disables regularization.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional scipy.optimize.lsq_linear iteration limit.",
    )
    parser.add_argument(
        "--liang-barsky",
        type=Path,
        default=Path(__file__).resolve().with_name("liang_barsky.C"),
        help=(
            "Path to the C++ file defining liang_barsky_3d. By default, "
            "liang_barsky.C is expected beside this Python script."
        ),
    )
    parser.add_argument(
        "--outname",
        type=Path,
        default=None,
        help="Output map file. A default name is used when omitted.",
    )
    return parser.parse_args()


def load_liang_barsky(source_path):
    """Load the C++ line-box intersection function and its PyROOT wrapper."""
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Liang-Barsky source not found: {source_path}\n"
            "Place liang_barsky.C beside this script or pass "
            "--liang-barsky /full/path/to/liang_barsky.C"
        )

    # Include the source directly in Cling rather than using ROOT's `.L` command.
    # This avoids path-parsing problems and ensures the function declaration is
    # visible when the Python-friendly wrapper is compiled.
    include_path = str(source_path).replace("\\", "\\\\").replace('"', '\\"')
    declaration = f"""
        #include <vector>
        #include "{include_path}"

        std::vector<double> liang_barsky_3d_py(
            double y0, double y1,
            double z0, double z1,
            double x0, double x1,
            double ey, double ez, double ex,
            double xy, double xz, double xx
        ) {{
            auto result = liang_barsky_3d(
                y0, y1, z0, z1, x0, x1,
                ey, ez, ex, xy, xz, xx
            );
            return std::vector<double>(result.begin(), result.end());
        }}
    """

    if not ROOT.gInterpreter.Declare(declaration):
        raise RuntimeError(f"Failed to compile Liang-Barsky source: {source_path}")


# def build_angelo2_file_list(data_dir, first_run, num_files):
#     """Construct the Angelo2 filenames used by this example."""
#     return [
#         data_dir / f"angelo2_run{run}_rsl100_abs20_500evts.root"
#         for run in range(first_run, first_run + num_files)
#     ]


# def build_chains(file_paths):
#     trajectory_chain = ROOT.TChain(TRAJECTORY_TREE)
#     photon_chain = ROOT.TChain(PHOTON_TREE)

#     missing = [path for path in file_paths if not path.is_file()]
#     if missing:
#         shown = "\n".join(f"  {path}" for path in missing[:5])
#         extra = f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else ""
#         raise FileNotFoundError(f"Missing input files:\n{shown}{extra}")

#     # The files must be added to both chains in the same order because the
#     # TChain tree number is used to distinguish repeated event IDs by file.
#     for path in file_paths:
#         trajectory_chain.Add(str(path))
#         photon_chain.Add(str(path))

#     if trajectory_chain.GetEntries() == 0 or photon_chain.GetEntries() == 0:
#         raise RuntimeError("One or both input trees contain no entries")

#     return trajectory_chain, photon_chain


# def count_detected_photons(photon_chain):
#     """Count selected detected photons for each file-local event."""
#     event_id = array("i", [0])
#     op_channel = array("i", [0])
#     photon_chain.SetBranchAddress("EventID", event_id)
#     photon_chain.SetBranchAddress("OpChannel", op_channel)

#     counts = {}
#     for entry in range(photon_chain.GetEntries()):
#         photon_chain.GetEntry(entry)
#         if op_channel[0] not in SELECTED_OP_CHANNELS:
#             continue

#         key = (photon_chain.GetTreeNumber(), event_id[0])
#         counts[key] = counts.get(key, 0) + 1

#     return counts


## FOR HDF5 FILES
def read_tracks_info_file(trk_info_file_path):
    """Open the h5 file with all track information, print track counts and selection criterion, return table with selected tracks"""
    ## Open the h5 file and access the tables within
    trk_info_file = tab.open_file(trk_info_file_path, mode="r")
    trk_info_table = trk_info_file.root.output
    trk_ct_table = trk_info_file.root.track_counts

    print("Total track counts across processed data:")
    for category in trk_ct_table.colnames:
        if category=="filename":
            continue
        ct_col = trk_ct_table.col(category) #If the column has subcolumns (is nested), table.col() returns the column as a structured np array with the subcolumns as fields. 
        paired = category not in ("all_tracks_ct", "a_crossers_ct", "c_crossers_ct") # the "stitched_c_crossers_ct" are paired.
        print(f"{category:<25} : {ct_col['all'].sum()/(2 if paired else 1):<10}{('pairs' if paired else ''):<6} --> with pds info: {ct_col['pds_cluster_matched'].sum()/(2 if paired else 1):<10}{('pairs' if paired else ''):<6}")
    print()
    print(f"Track selection criterion used:        {trk_info_table.attrs.selection_condition}")
    print(f"Total number of selected trajectories: {trk_info_table.nrows}")

    return trk_info_table 


def get_trajectory_start_end(trajectory_table_row : tab.tableextension.Row):
    """Get the start and end points of each trajectory from the hdf5 table"""
    reco_track_point_bounds = DETECTOR_BOUNDS.copy()
    reco_track_point_bounds['x'] = (-345, 345) # The (lardon z) (larsoft x) values are computed from e drift times referenced to estimated t0 (actual track-crossing time). It should be impossible to get a (larsoft) x significantly outside active volume unless t0 estimation is wrong (if the first e- reached the anode "before the track even crossed"). 
    # Maybe the y and z bounds should also be stricter --> how does LARDON determine x_ini/end and y_ini/end? Would it be reasonable for these values to be outside detector volume?
    # LArSoft (x,y,z) <---> LARDON (z,x,y)
    coords = {}
    coords["x_ini"], coords["y_ini"], coords["z_ini"] = trajectory_table_row['z_ini_corrected'], trajectory_table_row['x_ini'], trajectory_table_row['y_ini']  
    coords["x_end"], coords["y_end"], coords["z_end"] = trajectory_table_row['z_end_corrected'], trajectory_table_row['x_end'], trajectory_table_row['y_end']  

    is_paired = (trajectory_table_row["cathode_crosser_ID"] != -1)
    if is_paired:
        # The "cc_"-labelled tracks are supposed to be the ones in the upper volume.  
        coords["cc_x_ini"], coords["cc_y_ini"], coords["cc_z_ini"] = trajectory_table_row['cc_z_ini_corrected'], trajectory_table_row['cc_x_ini'], trajectory_table_row['cc_y_ini'] 
        coords["cc_x_end"], coords["cc_y_end"], coords["cc_z_end"] = trajectory_table_row['cc_z_end_corrected'], trajectory_table_row['cc_x_end'], trajectory_table_row['cc_y_end'] 

        ## Perform "quality control" tests
        # If any of the four points lie outside the allowed bounds, exit.
        for ax, bounds in reco_track_point_bounds.items():
            if np.any( [ ((coords[k] < bounds[0]) or (coords[k] > bounds[1])) for k in (f"cc_{ax}_ini",f"cc_{ax}_end",f"{ax}_ini",f"{ax}_end") ] ):
                return None

        # If the "cc_" (upper volume) track starts _below_ the start of the paired *lower* volume track (or ends below its end), something is wrong.
        # Basically if one track completely overlaps vertically with the other or the tracks are "on the wrong side", move on.    
        if (coords["cc_x_ini"] < coords["x_ini"]) or (coords["cc_x_end"] < coords["x_end"]):
            return None
        # Note: at this point the cc_ini --> end distance ought to be the largest among the four points involved. Can check this externally 
        start = np.array([ coords['cc_x_ini'], coords['cc_y_ini'], coords['cc_z_ini'] ])
        end   = np.array([ coords['x_end'],    coords['y_end'],    coords['z_end'] ])
        return start, end
    else:
        # If any of the four points lie outside the allowed bounds, exit.
        for ax, bounds in reco_track_point_bounds.keys():
            if np.any( [ ((coords[k] < bounds[0]) or (coords[k] > bounds[1])) for k in (f"{ax}_ini",f"{ax}_end") ] ):
                return None

        start = np.array([ coords['x_ini'], coords['y_ini'], coords['z_ini'] ])
        end   = np.array([ coords['x_end'], coords['y_end'], coords['z_end'] ])
        return start, end


def project_segment_to_detector_bounds(start, end, flipped=False):
    """Project a trajectory's start and end points to the detector bounds"""
    start, end = np.array(start), np.array(end) # just in case
    axes = ("x", "y", "z")
    # First try projecting to x boundary plane, then y boundary plane, then z boundary plane.
    for i, ax in enumerate(axes):
        planes = DETECTOR_BOUNDS[ax]
        end_target_plane = planes[0] if end[i] < start[i] else planes[1]
        tentative_proj_end = end + ((end_target_plane - end[i])/(end[i] - start[i]))*(end - start)

        # if the projection onto the x(y,z) boundary plane falls outside the y(z,x) or z(x,y) boundaries, try projecting onto another boundary. One of them has to work. 
        bounded1 = DETECTOR_BOUNDS[axes[i-1]][0] <= tentative_proj_end[i-1] <= DETECTOR_BOUNDS[axes[i-1]][1] 
        bounded2 = DETECTOR_BOUNDS[axes[i-2]][0] <= tentative_proj_end[i-2] <= DETECTOR_BOUNDS[axes[i-2]][1]
        if (not bounded1) or (not bounded2):
            continue
        else:
            proj_end = tentative_proj_end
    if not flipped:
        proj_start = project_segment_to_detector_bounds(end, start, flipped=True)
        return proj_start, proj_end
    else:
        return proj_end
#################

def build_voxel_bounds(ndiv):
    return (
        np.linspace(*DETECTOR_BOUNDS["x"], ndiv + 1),
        np.linspace(*DETECTOR_BOUNDS["y"], ndiv + 1),
        np.linspace(*DETECTOR_BOUNDS["z"], ndiv + 1),
    )


def voxel_index(ix, iy, iz, ndiv):
    """Flatten a voxel index with y varying fastest, matching the map output."""
    return (iz * ndiv + ix) * ndiv + iy


def fill_track_row(
    row_index,
    entry_point,
    exit_point,
    bounds,
    ndiv,
    rows,
    cols,
    values,
):
    """Add one track's nonzero voxel coefficients to sparse COO triplets."""
    xbounds, ybounds, zbounds = bounds

    for iz in range(ndiv):
        for ix in range(ndiv):
            for iy in range(ndiv):
                hit = ROOT.liang_barsky_3d_py(
                    ybounds[iy], ybounds[iy + 1],
                    zbounds[iz], zbounds[iz + 1],
                    xbounds[ix], xbounds[ix + 1],
                    entry_point[1], entry_point[2], entry_point[0],
                    exit_point[1], exit_point[2], exit_point[0],
                )

                if all(hit[index] == NO_INTERSECTION for index in range(6)):
                    continue

                dx = hit[3] - hit[0]
                dy = hit[4] - hit[1]
                dz = hit[5] - hit[2]
                path_length = math.sqrt(dx * dx + dy * dy + dz * dz)
                coefficient = MIP_ENERGY_LOSS * path_length

                if coefficient > 0.0:   # if it got to this point, wouldn't this be always true?
                    rows.append(row_index)
                    cols.append(voxel_index(ix, iy, iz, ndiv))
                    values.append(coefficient)


# def build_linear_system(trajectory_chain, photon_counts, ndiv, num_events):
def build_linear_system(trajectory_table, ndiv, num_events):
    """Build the sparse physical system A x = b."""
    # exit_point = array("f", [0.0, 0.0, 0.0])
    # entry_point = array("f", [0.0, 0.0, 0.0])
    # event_id = array("i", [0])
    # trajectory_chain.SetBranchAddress("CRT_ExitPoint", exit_point)
    # trajectory_chain.SetBranchAddress("CRT_EntryPoint", entry_point)
    # trajectory_chain.SetBranchAddress("Event", event_id)    

    # events_to_use = min(num_events, trajectory_chain.GetEntries())
    max_events_to_use = min(num_events, trajectory_table.nrows)
    bounds = build_voxel_bounds(ndiv)
    rows, cols, values = [], [], []
    # b = np.zeros(events_to_use, dtype=np.float64)
    b = []

    tracks_used = 0 # necessary since we don't know how many tracks will end up used. 
    # for row_index in range(events_to_use):
    for row_index, table_row in enumerate(trajectory_table.iterrows(start=0, stop=max_events_to_use)):
        # trajectory_chain.GetEntry(row_index)
        # key = (trajectory_chain.GetTreeNumber(), event_id[0])
        # b[row_index] = photon_counts.get(key, 0)

        start_end_points = get_trajectory_start_end(table_row)
        if start_end_points is None:
            continue  # If this trajectory is poor quality (points out of bounds, major stitching misalignment), skip it.

        b.append(table_row['total_npe']) # if this were above, b would get entries for rejected tracks and the shapes would be out of sync.
        entry_point, exit_point = project_segment_to_detector_bounds(*start_end_points)

        fill_track_row(
            # row_index,
            tracks_used, # the final matrix should have as many rows as tracks that were actually used; row_index will run ahead of tracks_used as tracks are rejected, so if we add row_index to rows we will go out of the matrix bounds.
            entry_point,
            exit_point,
            bounds,
            ndiv,
            rows,
            cols,
            values,
        )
        tracks_used += 1 # append it here so that tracks_used accurately reflects a 0-referenced index of the final matrix.

        # if (row_index + 1) % 1000 == 0 or row_index + 1 == events_to_use:
        #     print(f"Built {row_index + 1}/{events_to_use} event rows")
        if (tracks_used) % 1000 == 0 or row_index + 1 == max_events_to_use:
            print(f"Built {tracks_used} event rows")

    print(f"Out of {max_events_to_use} trajectories from the table, {tracks_used} were used\n")
    A = csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(cols, dtype=np.int32),
            ),
        ),
        # shape=(events_to_use, ndiv**3),
        shape=(tracks_used, ndiv**3),       # the final matrix should have as many rows as tracks that were actually used
    )
    return A, np.array(b)


def build_l2_smoothness_matrix(ndiv):
    """Build L so ||Lx||^2 sums all 6-neighbor voxel differences once."""
    rows, cols, values = [], [], []
    edge = 0

    for iz in range(ndiv):
        for ix in range(ndiv):
            for iy in range(ndiv):
                current = voxel_index(ix, iy, iz, ndiv)
                neighbors = []
                if ix + 1 < ndiv:
                    neighbors.append(voxel_index(ix + 1, iy, iz, ndiv))
                if iy + 1 < ndiv:
                    neighbors.append(voxel_index(ix, iy + 1, iz, ndiv))
                if iz + 1 < ndiv:
                    neighbors.append(voxel_index(ix, iy, iz + 1, ndiv))

                for neighbor in neighbors:
                    rows.extend((edge, edge))       # list.extend(iterable) appends all items from iterable to end of list
                    cols.extend((current, neighbor))
                    values.extend((1.0, -1.0))
                    edge += 1

    return csr_matrix(
        (values, (rows, cols)),
        shape=(edge, ndiv**3),
        dtype=np.float64,
    )


def add_smoothness(A, b, ndiv, lambda_smooth):
    """Augment [A, b] with sqrt(lambda) L and a zero target."""
    if lambda_smooth == 0.0:
        return A, b

    L = build_l2_smoothness_matrix(ndiv)
    A_solve = vstack((A, math.sqrt(lambda_smooth) * L), format="csr")
    b_solve = np.concatenate((b, np.zeros(L.shape[0], dtype=np.float64)))
    return A_solve, b_solve


def solve(A, b, max_iter):
    result = lsq_linear(
        A,
        b,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
        max_iter=max_iter,
        verbose=1,
    )
    if not result.success:
        raise RuntimeError(f"Least-squares solver failed: {result.message}")
    return result.x


def output_path(args):
    if args.outname is not None:
        return args.outname

    if args.lambda_smooth == 0.0:
        tag = "unregularized"
    else:
        value = f"{args.lambda_smooth:.6g}".replace("-", "m").replace("+", "") # why is this here if lambda_smooth has to be non-negative ??
        tag = f"lambda{value.replace('.', 'p')}"

    # return Path(
    #     f"solutions/angelo2_{args.ndiv}x{args.ndiv}x{args.ndiv}_"
    #     f"{args.num_events}evts_{tag}.txt"
    # )
    return Path(
        f"solutions/{args.ndiv}x{args.ndiv}x{args.ndiv}_{tag}.txt"
    )


def write_solution(path, ndiv, solution):
    """Write ndiv followed by one flattened voxel value per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write(f"{ndiv}\n")
        for value in solution:
            output.write(f"{value:.17g}\n")


def main():
    args = parse_args()
    # if args.ndiv <= 0 or args.num_events <= 0 or args.num_files <= 0:
    #     raise ValueError("--ndiv, --num-events, and --num-files must be positive")
    # if args.first_run < 0 or args.lambda_smooth < 0.0:
    #     raise ValueError("--first-run and --lambda-smooth must be non-negative")
    if args.ndiv <= 0 or args.num_events <= 0:
        raise ValueError("--ndiv and --num-events must be positive")
    if args.lambda_smooth < 0.0:
        raise ValueError("--lambda-smooth must be non-negative")

    load_liang_barsky(args.liang_barsky)
    # files = build_angelo2_file_list(args.data_dir, args.first_run, args.num_files)
    # trajectory_chain, photon_chain = build_chains(files)

    # print(f"Trajectory entries: {trajectory_chain.GetEntries()}")
    # print(f"Detected-photon entries: {photon_chain.GetEntries()}")

    # photon_counts = count_detected_photons(photon_chain)

    trajectory_table = read_tracks_info_file(args.data_file)

    # A, b = build_linear_system(
    #     trajectory_chain,
    #     photon_counts,
    #     args.ndiv,
    #     args.num_events,
    # )
    A, b = build_linear_system(
        trajectory_table,
        args.ndiv,
        args.num_events,
    )
    print(f"Sparse system: shape={A.shape}, nonzeros={A.nnz}")

    A_solve, b_solve = add_smoothness(A, b, args.ndiv, args.lambda_smooth)
    max_iter = args.max_iter if args.max_iter is not None else 10 * A.shape[0]
    solution = solve(A_solve, b_solve, max_iter)

    destination = output_path(args)
    write_solution(destination, args.ndiv, solution)
    print(f"Wrote solution to {destination}")


if __name__ == "__main__":
    main()
