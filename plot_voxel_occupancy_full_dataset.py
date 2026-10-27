#!/usr/bin/env python3
"""
Voxel occupancy diagnostics for PDS calibration CRT track samples.

Computes, for a selected track sample:
  - per-voxel occupancy N_j = number of tracks crossing voxel j
  - per-voxel total path length L_j
  - slice summaries along x, y, and z
  - projection plots and optional per-slice plots in all three axes

Default indexing matches pds_cal_gen_scipy_sparse_cli.py:
    idx = iy + ndiv * (ix + ndiv * iz)

Full angelo2_config example:
  python3 plot_voxel_occupancy_full_dataset.py \
    --ndiv 16 \
    --sample-mode all \
    --outdir occupancy_16x16_full_angelo2 \
    --angelo-first-run 0 \
    --angelo-last-run 119 \
    --angelo-template "/home/ajheindel/Software/pds_cal/data/angelo2_config/angelo2_run{i}_rsl100_abs20_500evts.root" \
    --make-slice-plots \
    --log-occupancy
"""

from __future__ import annotations

import argparse
import csv
import math
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tables as tab
from read_lardon_tracks_h5tools import get_trajectory_start_end, project_segment_to_detector_bounds

try:
    import ROOT  # type: ignore
except Exception as exc:  # pragma: no cover
    ROOT = None
    ROOT_IMPORT_ERROR = exc
else:
    ROOT_IMPORT_ERROR = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute voxel occupancy and slice-by-slice occupancy diagnostics for CRT tracks."
    )

    p.add_argument("--ndiv", type=int, required=True)

    p.add_argument("--xmin", type=float, default=-400.0)
    p.add_argument("--xmax", type=float, default=400.0)
    p.add_argument("--ymin", type=float, default=-425.0)
    p.add_argument("--ymax", type=float, default=425.0)
    p.add_argument("--zmin", type=float, default=-300.0)
    p.add_argument("--zmax", type=float, default=600.0)

    p.add_argument("--outdir", default="voxel_occupancy_diagnostics")
    p.add_argument("--track-info-file", default="out_39273_a_ac_ca_aca.h5")
    # p.add_argument("--track-tree", default="TrajCut/CRTTrajectoryCut")

    # p.add_argument("--full-config-runs", type=int, default=0)
    # p.add_argument("--full-config-template",
    #                default="/home/ajheindel/Software/pds_cal/data/full_config/full_run{i}_rsl100_abs20_500evts.root")
    # p.add_argument("--angelo-first-run", type=int, default=0)
    # p.add_argument("--angelo-last-run", type=int, default=119)
    # p.add_argument("--angelo-template",
    #                default="/home/ajheindel/Software/pds_cal/data/angelo2_config/angelo2_run{i}_rsl100_abs20_500evts.root")

    p.add_argument("--interior-exclusion", type=int, default=-1,
        help="Margin in cm with which to exclude tracks whose start/end points lie in active volume interior. -1 for no exclusion.",
    )
    p.add_argument("--y-exclusion", type=int, default=-1,
        help="Exclude all tracks whose start/end point abs(y-coord) is this value or less in cm. -1 for no exclusion.",
    )

    p.add_argument("--sample-mode", choices=["all", "sequential", "random"], default="all")
    p.add_argument("--num-events", type=int, default=0,
                   help="Number of eligible volume-crossing tracks for sequential/random mode. Use 0 with --sample-mode all.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--shuffle-row-order", action="store_true",
                   help="For random mode, keep random row order instead of sorting selected entries.")
    p.add_argument("--resample", action="store_true", 
                   help="Attempts to subsample the set of tracks to achieve a more uniform voxel-occupancy map. Creates diagnostic plots for that resample and returns the resampled-trackIDs as a list.")

    p.add_argument("--index-order", choices=["solver_zxy", "standard_xyz"], default="standard_xyz")
    p.add_argument("--progress-every", type=int, default=5000)

    p.add_argument("--make-slice-plots", action="store_true",
                   help="Write one heatmap per slice along x, y, and z.")
    p.add_argument("--plot-path", action="store_true",
                   help="Also make total-path-length slice plots. Occupancy slice plots are always made.")
    p.add_argument("--max-slice-plots-per-axis", type=int, default=0,
                   help="If >0, only plot this many central slices per axis.")
    p.add_argument("--plot-format", choices=["png", "pdf", "both"], default="both")
    p.add_argument("--occupancy-vmax", type=float, default=0.0) # the occupancy value mapped to the upper end of the colormap in the occupancy heatmaps
    p.add_argument("--path-vmax", type=float, default=0.0)      # the path-length value mapped to the upper end of the colormap in the path-length heatmaps
    p.add_argument("--log-occupancy", action="store_true",
                   help="Also save log10(occupancy+1) maps/projections.")

    return p.parse_args()


# def build_file_list(args: argparse.Namespace) -> List[str]:
#     files: List[str] = []

#     for i in range(args.full_config_runs):
#         files.append(args.full_config_template.format(i=i))

#     if args.angelo_last_run >= args.angelo_first_run:
#         for i in range(args.angelo_first_run, args.angelo_last_run + 1):
#             files.append(args.angelo_template.format(i=i))

#     missing = [f for f in files if not Path(f).exists()]
#     if missing:
#         print("ERROR: missing input ROOT files:")
#         for f in missing[:20]:
#             print(f"  {f}")
#         if len(missing) > 20:
#             print(f"  ... and {len(missing) - 20} more")
#         raise SystemExit(1)

#     if not files:
#         raise SystemExit("ERROR: no ROOT files configured.")

#     return files


# def setup_tchain(args: argparse.Namespace):
#     if ROOT is None:
#         raise RuntimeError(f"Could not import ROOT/PyROOT: {ROOT_IMPORT_ERROR}")

#     ROOT.gROOT.SetBatch(True)
#     t = ROOT.TChain(args.track_tree)
#     files = build_file_list(args)

#     print(f"Adding {len(files)} files to {args.track_tree}")
#     for f in files:
#         t.Add(f)

#     print(f"TChain entries: {int(t.GetEntries())}")
#     return t, files


def build_bounds(args: argparse.Namespace, ndiv: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(args.xmin, args.xmax, ndiv + 1),
        np.linspace(args.ymin, args.ymax, ndiv + 1),
        np.linspace(args.zmin, args.zmax, ndiv + 1),
    )


def flat_index(ix: int, iy: int, iz: int, ndiv: int, order: str) -> int:
    if order == "solver_zxy":
        return iy + ndiv * (ix + ndiv * iz)
    if order == "standard_xyz":
        return ix + ndiv * (iy + ndiv * iz)
    raise ValueError(order)


def inv_flat_index(idx: int, ndiv: int, order: str) -> Tuple[int, int, int]:
    if order == "solver_zxy":
        iz = idx // (ndiv * ndiv)
        rem = idx - iz * ndiv * ndiv
        ix = rem // ndiv
        iy = rem - ix * ndiv
        return ix, iy, iz
    if order == "standard_xyz":
        iz = idx // (ndiv * ndiv)
        rem = idx - iz * ndiv * ndiv
        iy = rem // ndiv
        ix = rem - iy * ndiv
        return ix, iy, iz
    raise ValueError(order)


def voxel_center(ix: int, iy: int, iz: int,
                 xbounds: np.ndarray, ybounds: np.ndarray, zbounds: np.ndarray) -> Tuple[float, float, float]:
    return (
        0.5 * (xbounds[ix] + xbounds[ix + 1]),
        0.5 * (ybounds[iy] + ybounds[iy + 1]),
        0.5 * (zbounds[iz] + zbounds[iz + 1]),
    )


def clip_segment_to_box(p0: np.ndarray, p1: np.ndarray,
                        mins: np.ndarray, maxs: np.ndarray) -> Optional[Tuple[float, float]]:
    d = p1 - p0
    t0 = 0.0
    t1 = 1.0

    for a in range(3):
        if abs(d[a]) < 1.0e-14:
            if p0[a] < mins[a] or p0[a] > maxs[a]:
                return None
            continue

        inv = 1.0 / d[a]
        ta = (mins[a] - p0[a]) * inv
        tb = (maxs[a] - p0[a]) * inv
        if ta > tb:
            ta, tb = tb, ta

        t0 = max(t0, ta)
        t1 = min(t1, tb)

        if t1 <= t0:
            return None

    return t0, t1


def segment_voxel_lengths(
    p0_tuple: Sequence[float],
    p1_tuple: Sequence[float],
    xbounds: np.ndarray,
    ybounds: np.ndarray,
    zbounds: np.ndarray,
    ndiv: int,
    order: str,
) -> Dict[int, float]:
    p0 = np.asarray(p0_tuple, dtype=float)
    p1 = np.asarray(p1_tuple, dtype=float)
    mins = np.asarray([xbounds[0], ybounds[0], zbounds[0]], dtype=float)
    maxs = np.asarray([xbounds[-1], ybounds[-1], zbounds[-1]], dtype=float)

    clipped = clip_segment_to_box(p0, p1, mins, maxs)
    if clipped is None:
        return {}

    t_start, t_stop = clipped
    d = p1 - p0
    seg_len = float(np.linalg.norm(d))
    if seg_len <= 0.0:
        return {}

    svals = [t_start, t_stop]
    for a, bounds in enumerate([xbounds, ybounds, zbounds]):
        if abs(d[a]) < 1.0e-14:
            continue
        for boundary in bounds[1:-1]:
            s = (boundary - p0[a]) / d[a]
            if t_start + 1.0e-12 < s < t_stop - 1.0e-12:
                svals.append(float(s))

    svals = sorted(svals)
    unique_s: List[float] = []
    for s in svals:
        if not unique_s or abs(s - unique_s[-1]) > 1.0e-10:
            unique_s.append(s)

    widths = np.asarray([
        (xbounds[-1] - xbounds[0]) / ndiv,
        (ybounds[-1] - ybounds[0]) / ndiv,
        (zbounds[-1] - zbounds[0]) / ndiv,
    ])

    out: Dict[int, float] = {}
    for a, b in zip(unique_s[:-1], unique_s[1:]):
        if b <= a:
            continue

        mid = 0.5 * (a + b)
        pos = p0 + mid * d

        ix = int(math.floor((pos[0] - xbounds[0]) / widths[0]))
        iy = int(math.floor((pos[1] - ybounds[0]) / widths[1]))
        iz = int(math.floor((pos[2] - zbounds[0]) / widths[2]))

        ix = min(max(ix, 0), ndiv - 1)
        iy = min(max(iy, 0), ndiv - 1)
        iz = min(max(iz, 0), ndiv - 1)

        idx = flat_index(ix, iy, iz, ndiv, order)
        dl = (b - a) * seg_len
        if dl > 0.0:
            out[idx] = out.get(idx, 0.0) + dl

    return out


# def select_entries(tchain, args: argparse.Namespace,
def select_entries(trajectory_table, args: argparse.Namespace,
                #    entry_points: array,
                #    exit_points: array,
                   xbounds: np.ndarray,
                   ybounds: np.ndarray,
                   zbounds: np.ndarray) -> List[int]:
    """Returns a list of indices of entries of the trajectory table corresponding to tracks falling within the specified volume, optionally subsamples said list."""
    # nentries = int(tchain.GetEntries())
    nentries = trajectory_table.nrows
    mins = np.asarray([xbounds[0], ybounds[0], zbounds[0]], dtype=float)
    maxs = np.asarray([xbounds[-1], ybounds[-1], zbounds[-1]], dtype=float)

    eligible: List[int] = []
    print("Finding entries crossing solved volume...")

    for entry in range(nentries):
        if args.progress_every > 0 and entry > 0 and entry % args.progress_every == 0:
            print(f"  checked {entry} / {nentries}; eligible={len(eligible)}")

        # tchain.GetEntry(entry)
        track = trajectory_table[entry]
        if args.interior_exclusion < 0 and args.y_exclusion < 0:
            start_end_tuple = get_trajectory_start_end(track)
        else:
            start_end_tuple = get_trajectory_start_end(track, args.interior_exclusion, args.y_exclusion)
    
        if start_end_tuple is None:
            continue  # If this trajectory is poor quality (points out of bounds, major stitching misalignment), skip it.

        ## We are treating the data-tracks projected to the actual detector bounds (not the bounds passed into the arguments of this script) as the "real" start and end points. Whether these extended tracks fall within a potentially smaller volume specified in the arguments is then what the clip_segments_to_box function is for. 
        ## After all the entry_points and exit_points that were being used for p0 and p1 below in the simulated-tracks case were the values in the CRT_EntryPoint and CRT_ExitPoint fields of the TChain -- i.e they were even past the solver's detector bounds (the "CRTs" were at y = +-600) 
        entry_point, exit_point = project_segment_to_detector_bounds(*start_end_tuple)
        
        # p0 = np.asarray([entry_points[0], entry_points[1], entry_points[2]], dtype=float)
        # p1 = np.asarray([exit_points[0], exit_points[1], exit_points[2]], dtype=float)
        p0 = np.asarray([entry_point[0], entry_point[1], entry_point[2]], dtype=float)
        p1 = np.asarray([exit_point[0], exit_point[1], exit_point[2]], dtype=float)

        if clip_segment_to_box(p0, p1, mins, maxs) is not None:
            eligible.append(entry)

    print(f"Eligible volume-crossing entries: {len(eligible)} / {nentries}")

    if args.sample_mode == "all":
        return eligible

    if args.num_events <= 0:
        raise SystemExit("ERROR: --num-events must be >0 for sequential/random sample mode.")

    n_to_use = min(args.num_events, len(eligible))

    if args.sample_mode == "sequential":
        return eligible[:n_to_use]

    rng = np.random.default_rng(args.seed)
    selected = rng.permutation(np.asarray(eligible, dtype=np.int64))[:n_to_use].astype(int).tolist()
    if not args.shuffle_row_order:
        selected = sorted(selected)
    return selected


def save_formats(fig, path_no_ext: Path, args: argparse.Namespace, dpi: int = 160) -> None:
    if args.plot_format in ("png", "both"):
        fig.savefig(path_no_ext.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if args.plot_format in ("pdf", "both"):
        fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")


def plot_heatmap(data2d: np.ndarray,
                 extent: Tuple[float, float, float, float],
                 xlabel: str,
                 ylabel: str,
                 title: str,
                 cbar_label: str,
                 outpath: Path,
                 args: argparse.Namespace,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(data2d, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_formats(fig, outpath, args)
    plt.close(fig)


def central_slice_indices(ndiv: int, max_slices: int) -> List[int]:
    if max_slices <= 0 or max_slices >= ndiv:
        return list(range(ndiv))
    center = (ndiv - 1) / 2.0
    order = sorted(range(ndiv), key=lambda i: abs(i - center))
    return sorted(order[:max_slices])


def write_outputs(args: argparse.Namespace,
                  outdir: Path,
                  occupancy: np.ndarray,
                  total_path: np.ndarray,
                  selected_entries: Sequence[int],
                #   files: Sequence[str],
                  dirs_arr: np.ndarray,
                  track_lengths_arr: np.ndarray) -> None:
    ndiv = args.ndiv
    xbounds, ybounds, zbounds = build_bounds(args, ndiv)
    crossed = occupancy > 0

    # with (outdir / "input_files.txt").open("w") as f:
    #     for i, path in enumerate(files):
    #         f.write(f"{i},{path}\n")

    with (outdir / "selected_entries.txt").open("w") as f:
        f.write("# row,entry\n")
        for row, entry in enumerate(selected_entries):
            f.write(f"{row},{entry}\n")

    mean_path = np.divide(total_path, occupancy, out=np.zeros_like(total_path), where=occupancy > 0)
    np.savez_compressed(
        outdir / "voxel_occupancy_maps.npz",
        occupancy=occupancy,
        total_path_cm=total_path,
        mean_path_per_crossing_cm=mean_path,
        selected_entries=np.asarray(selected_entries, dtype=np.int64),
        xbounds=xbounds,
        ybounds=ybounds,
        zbounds=zbounds,
        track_directions=dirs_arr,
        track_lengths_cm=track_lengths_arr,
    )

    with (outdir / "voxel_occupancy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "ix", "iy", "iz",
            "x_center_cm", "y_center_cm", "z_center_cm",
            "occupancy", "total_path_cm", "mean_path_per_crossing_cm",
        ])
        for iz in range(ndiv):
            for ix in range(ndiv):
                for iy in range(ndiv):
                    idx = flat_index(ix, iy, iz, ndiv, args.index_order)
                    occ = int(occupancy[ix, iy, iz])
                    path = float(total_path[ix, iy, iz])
                    xc, yc, zc = voxel_center(ix, iy, iz, xbounds, ybounds, zbounds)
                    w.writerow([idx, ix, iy, iz, xc, yc, zc, occ, path, path / occ if occ else 0.0])

    with (outdir / "slice_occupancy_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "axis", "slice_index", "slice_center_cm",
            "n_voxels", "n_crossed_voxels", "crossed_fraction",
            "occupancy_sum", "occupancy_mean_all", "occupancy_mean_crossed",
            "occupancy_median_crossed", "occupancy_max",
            "total_path_cm", "path_mean_all_cm", "path_mean_crossed_cm",
            "path_median_crossed_cm", "path_max_cm",
        ])
        for axis in ["x", "y", "z"]:
            for sidx in range(ndiv):
                if axis == "x":
                    occ2, path2 = occupancy[sidx, :, :], total_path[sidx, :, :]
                    center = 0.5 * (xbounds[sidx] + xbounds[sidx + 1])
                elif axis == "y":
                    occ2, path2 = occupancy[:, sidx, :], total_path[:, sidx, :]
                    center = 0.5 * (ybounds[sidx] + ybounds[sidx + 1])
                else:
                    occ2, path2 = occupancy[:, :, sidx], total_path[:, :, sidx]
                    center = 0.5 * (zbounds[sidx] + zbounds[sidx + 1])

                mask = occ2 > 0
                n_vox = occ2.size
                n_cross = int(np.count_nonzero(mask))
                w.writerow([
                    axis, sidx, center,
                    n_vox, n_cross, n_cross / n_vox if n_vox else 0.0,
                    int(occ2.sum()), float(occ2.mean()), float(occ2[mask].mean()) if n_cross else 0.0,
                    float(np.median(occ2[mask])) if n_cross else 0.0, int(occ2.max()),
                    float(path2.sum()), float(path2.mean()), float(path2[mask].mean()) if n_cross else 0.0,
                    float(np.median(path2[mask])) if n_cross else 0.0, float(path2.max()),
                ])

    with (outdir / "occupancy_summary.txt").open("w") as f:
        f.write("Voxel occupancy diagnostics\n")
        f.write("===========================\n")
        f.write(f"ndiv: {ndiv}\n")
        f.write(f"Input filename: {args.track_info_file}\n")
        f.write(f"index_order: {args.index_order}\n")
        f.write(f"sample_mode: {args.sample_mode}\n")
        f.write(f"num_events argument: {args.num_events}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"selected tracks: {len(selected_entries)}\n")
        f.write(f"nonzero occupancy voxels: {int(np.count_nonzero(crossed))} / {ndiv**3}\n")
        f.write(f"crossed fraction: {float(np.count_nonzero(crossed)) / (ndiv**3):.6g}\n")
        f.write(f"occupancy sum over voxels: {int(occupancy.sum())}\n")
        f.write(f"occupancy max: {int(occupancy.max())}\n")
        f.write(f"occupancy mean all voxels: {float(occupancy.mean()):.6g}\n")
        f.write(f"occupancy mean crossed voxels: {float(occupancy[crossed].mean()) if np.any(crossed) else 0.0:.6g}\n")
        f.write(f"occupancy median crossed voxels: {float(np.median(occupancy[crossed])) if np.any(crossed) else 0.0:.6g}\n")
        f.write(f"occupancy p10 crossed voxels: {float(np.percentile(occupancy[crossed], 10)) if np.any(crossed) else 0.0:.6g}\n")
        f.write(f"occupancy p90 crossed voxels: {float(np.percentile(occupancy[crossed], 90)) if np.any(crossed) else 0.0:.6g}\n")
        f.write(f"total path length in volume: {float(total_path.sum()):.6g} cm\n")
        f.write(f"mean path all voxels: {float(total_path.mean()):.6g} cm\n")
        f.write(f"mean path crossed voxels: {float(total_path[crossed].mean()) if np.any(crossed) else 0.0:.6g} cm\n")
        f.write(f"median path crossed voxels: {float(np.median(total_path[crossed])) if np.any(crossed) else 0.0:.6g} cm\n")
        if dirs_arr.size:
            f.write("\nTrack direction summary\n")
            f.write("-----------------------\n")
            f.write(f"mean ux,uy,uz: {dirs_arr[:,0].mean():.6g}, {dirs_arr[:,1].mean():.6g}, {dirs_arr[:,2].mean():.6g}\n")
            f.write(f"mean |ux|,|uy|,|uz|: {np.abs(dirs_arr[:,0]).mean():.6g}, {np.abs(dirs_arr[:,1]).mean():.6g}, {np.abs(dirs_arr[:,2]).mean():.6g}\n")
            f.write(f"median clipped track length: {np.median(track_lengths_arr):.6g} cm\n")
            f.write(f"mean clipped track length: {track_lengths_arr.mean():.6g} cm\n")

    if dirs_arr.size:
        with (outdir / "track_direction_summary.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["quantity", "value"])
            rows = [
                ("n_tracks", len(dirs_arr)),
                ("mean_ux", dirs_arr[:, 0].mean()),
                ("mean_uy", dirs_arr[:, 1].mean()),
                ("mean_uz", dirs_arr[:, 2].mean()),
                ("mean_abs_ux", np.abs(dirs_arr[:, 0]).mean()),
                ("mean_abs_uy", np.abs(dirs_arr[:, 1]).mean()),
                ("mean_abs_uz", np.abs(dirs_arr[:, 2]).mean()),
                ("median_track_length_cm", np.median(track_lengths_arr)),
                ("mean_track_length_cm", track_lengths_arr.mean()),
            ]
            w.writerows(rows)


def make_projection_plots(args: argparse.Namespace,
                          outdir: Path,
                          occupancy: np.ndarray,
                          total_path: np.ndarray) -> None:
    projdir = outdir / "projections"
    projdir.mkdir(parents=True, exist_ok=True)
    occ_vmax = args.occupancy_vmax if args.occupancy_vmax > 0 else None
    path_vmax = args.path_vmax if args.path_vmax > 0 else None

    projections = [
        ("yx", occupancy.sum(axis=2), total_path.sum(axis=2),
         (args.ymin, args.ymax, args.xmin, args.xmax), "Y (cm)", "X (cm)", "sum over z"),
        ("zx", occupancy.sum(axis=1), total_path.sum(axis=1),
         (args.zmin, args.zmax, args.xmin, args.xmax), "Z (cm)", "X (cm)", "sum over y"),
        ("zy", occupancy.sum(axis=0), total_path.sum(axis=0),
         (args.zmin, args.zmax, args.ymin, args.ymax), "Z (cm)", "Y (cm)", "sum over x"),
    ]

    for name, occ2, path2, extent, xlabel, ylabel, collapse in projections:
        plot_heatmap(
            occ2, extent, xlabel, ylabel,
            f"Voxel occupancy projection ({collapse}), {name.upper()}",
            "track-voxel crossings", projdir / f"occupancy_projection_{name}", args,
            vmin=0, vmax=occ_vmax,
        )
        plot_heatmap(
            path2, extent, xlabel, ylabel,
            f"Total path-length projection ({collapse}), {name.upper()}",
            "total path length (cm)", projdir / f"total_path_projection_{name}", args,
            vmin=0, vmax=path_vmax,
        )
        if args.log_occupancy:
            plot_heatmap(
                np.log10(occ2 + 1.0), extent, xlabel, ylabel,
                f"log10(occupancy+1) projection ({collapse}), {name.upper()}",
                r"$\log_{10}(N+1)$", projdir / f"log_occupancy_projection_{name}", args,
                vmin=0, vmax=None,
            )


def make_slice_summary_plots(args: argparse.Namespace,
                             outdir: Path,
                             occupancy: np.ndarray,
                             total_path: np.ndarray) -> None:
    ndiv = args.ndiv
    xbounds, ybounds, zbounds = build_bounds(args, ndiv)
    plotdir = outdir / "slice_summary_plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    for axis, centers in [
        ("x", 0.5 * (xbounds[:-1] + xbounds[1:])),
        ("y", 0.5 * (ybounds[:-1] + ybounds[1:])),
        ("z", 0.5 * (zbounds[:-1] + zbounds[1:])),
    ]:
        occ_sum, occ_mean, occ_max, crossed_frac, path_sum = [], [], [], [], []
        for sidx in range(ndiv):
            if axis == "x":
                occ2, path2 = occupancy[sidx, :, :], total_path[sidx, :, :]
            elif axis == "y":
                occ2, path2 = occupancy[:, sidx, :], total_path[:, sidx, :]
            else:
                occ2, path2 = occupancy[:, :, sidx], total_path[:, :, sidx]
            occ_sum.append(float(occ2.sum()))
            occ_mean.append(float(occ2.mean()))
            occ_max.append(float(occ2.max()))
            crossed_frac.append(float(np.count_nonzero(occ2 > 0) / occ2.size))
            path_sum.append(float(path2.sum()))

        for vals, ylabel, name in [
            (occ_sum, "sum of voxel occupancies in slice", "occupancy_sum"),
            (occ_mean, "mean voxel occupancy in slice", "occupancy_mean"),
            (occ_max, "maximum voxel occupancy in slice", "occupancy_max"),
            (crossed_frac, "fraction of voxels crossed", "crossed_fraction"),
            (path_sum, "total path length in slice (cm)", "total_path_sum"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.plot(centers, vals, marker="o")
            ax.set_xlabel(f"{axis.upper()} slice center (cm)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} versus {axis.upper()} slice")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_formats(fig, plotdir / f"{name}_vs_{axis}_slice", args)
            plt.close(fig)


def make_slice_plots(args: argparse.Namespace,
                     outdir: Path,
                     occupancy: np.ndarray,
                     total_path: np.ndarray) -> None:
    ndiv = args.ndiv
    slicedir = outdir / "slices"
    slicedir.mkdir(parents=True, exist_ok=True)
    occ_vmax = args.occupancy_vmax if args.occupancy_vmax > 0 else None
    path_vmax = args.path_vmax if args.path_vmax > 0 else None
    indices = central_slice_indices(ndiv, args.max_slice_plots_per_axis)

    for axis in ["x", "y", "z"]:
        axisdir = slicedir / f"constant_{axis}"
        axisdir.mkdir(parents=True, exist_ok=True)
        for sidx in indices:
            if axis == "x":
                occ2, path2 = occupancy[sidx, :, :], total_path[sidx, :, :]
                extent = (args.zmin, args.zmax, args.ymin, args.ymax)
                xlabel, ylabel, plane = "Z (cm)", "Y (cm)", "YZ"
                center = args.xmin + (sidx + 0.5) * (args.xmax - args.xmin) / ndiv  # [(args.xmax - args.xmin) / ndiv] is the x-width of each voxel, so the left edge of the ith voxel becomes xmin + i*[(xmax-xmin)/ n] and its center would be 0.5*[(xmax-xmin)/ n] further 
            elif axis == "y":
                occ2, path2 = occupancy[:, sidx, :], total_path[:, sidx, :]
                extent = (args.zmin, args.zmax, args.xmin, args.xmax)
                xlabel, ylabel, plane = "Z (cm)", "X (cm)", "XZ"
                center = args.ymin + (sidx + 0.5) * (args.ymax - args.ymin) / ndiv
            else:
                occ2, path2 = occupancy[:, :, sidx], total_path[:, :, sidx]
                extent = (args.ymin, args.ymax, args.xmin, args.xmax)
                xlabel, ylabel, plane = "Y (cm)", "X (cm)", "YX"
                center = args.zmin + (sidx + 0.5) * (args.zmax - args.zmin) / ndiv

            plot_heatmap(
                occ2, extent, xlabel, ylabel,
                f"Occupancy, {plane} slice {axis}={sidx} ({center:.1f} cm)",
                "tracks crossing voxel", axisdir / f"occupancy_{plane}_slice_{axis}{sidx:02d}", args,
                vmin=0, vmax=occ_vmax,
            )
            if args.log_occupancy:
                plot_heatmap(
                    np.log10(occ2 + 1.0), extent, xlabel, ylabel,
                    f"log10(occupancy+1), {plane} slice {axis}={sidx} ({center:.1f} cm)",
                    r"$\log_{10}(N+1)$", axisdir / f"log_occupancy_{plane}_slice_{axis}{sidx:02d}", args,
                    vmin=0, vmax=None,
                )
            if args.plot_path:
                plot_heatmap(
                    path2, extent, xlabel, ylabel,
                    f"Total path length, {plane} slice {axis}={sidx} ({center:.1f} cm)",
                    "total path length (cm)", axisdir / f"total_path_{plane}_slice_{axis}{sidx:02d}", args,
                    vmin=0, vmax=path_vmax,
                )


def make_track_direction_plots(args: argparse.Namespace,
                               outdir: Path,
                               dirs_arr: np.ndarray,
                               track_lengths_arr: np.ndarray) -> None:
    if dirs_arr.size == 0:
        return
    plotdir = outdir / "track_direction_plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    for name, col in [("ux", 0), ("uy", 1), ("uz", 2)]:
        for arr, suffix, xlabel in [
            (dirs_arr[:, col], name, name),
            (np.abs(dirs_arr[:, col]), f"abs_{name}", f"|{name}|"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.hist(arr, bins=60, histtype="step", linewidth=1.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("tracks")
            ax.set_title(f"Track direction component {xlabel}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_formats(fig, plotdir / f"track_direction_{suffix}", args)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(track_lengths_arr, bins=60, histtype="step", linewidth=1.8)
    ax.set_xlabel("clipped track length in solved volume (cm)")
    ax.set_ylabel("tracks")
    ax.set_title("Track path length through solved volume")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_formats(fig, plotdir / "track_length_in_volume", args)
    plt.close(fig)

####### TEST #######
def resample_balanced_occupancy(args, outdir, trajectory_table, selected_entries, occupancy, temp=None):
    ndiv = args.ndiv
    xbounds, ybounds, zbounds = build_bounds(args, ndiv)
    
    n_voxels_crossed = np.zeros( len(selected_entries), dtype=np.float64 )
    total_occ_of_voxels_crossed = np.zeros( len(selected_entries), dtype=np.float64 )
    crossing_info_list = {}
    for i, entry in enumerate(selected_entries):
        track = trajectory_table[entry]

        # the projected points are the equivalent of the "CRT" entry/exit points
        entry_point, exit_point = project_segment_to_detector_bounds(*get_trajectory_start_end(track)) # from select_entries, we know that track will pass the tests in get_trajectory_start_end. So it's kind of redundant to have them be done again, but hopefully it does not add too much runtime.  
        p0 = (float(entry_point[0]), float(entry_point[1]), float(entry_point[2]))
        p1 = (float(exit_point[0]), float(exit_point[1]), float(exit_point[2]))
    
        lengths = segment_voxel_lengths(p0, p1, xbounds, ybounds, zbounds, ndiv, args.index_order)
        if not lengths:
            continue   # this should never happen given the tracks are from selected_entries 
        crossing_info_list[entry] = (lengths, p0, p1)
    
        for idx in lengths.keys():
            # ix, iy, iz are the "coordinates" specifying a specific voxel in xyz space. args.index_order specifies the scheme by which the index idx in the flat voxel array of the voxel at (ix, iy, iz) is computed.   
            ix, iy, iz = inv_flat_index(idx, ndiv, args.index_order)
            n_voxels_crossed[i] += 1
            total_occ_of_voxels_crossed[i] += occupancy[ix, iy, iz]

    # Assign every track a weight based on the average occupancy across said track.
    occ_per_voxels_crossed = total_occ_of_voxels_crossed / n_voxels_crossed 
    # T = np.min(occ_per_voxels_crossed)
    T = np.mean(occ_per_voxels_crossed)
    # T = np.median(occ_per_voxels_crossed) # The median is much lower than the mean -- there are many high outliers which pull up mean but not median. 
    
    ## exponential
    # weights = np.exp( -(occ_per_voxels_crossed) / T )
    # weights = np.exp( -(occ_per_voxels_crossed - T) )
    
    ## ~1/x -- p=1 at min (47000/120000, CV = 0.75 when T = min)
    # weights = T / (occ_per_voxels_crossed)
    
    ## ~1/x**2 -- p=1 at min -- way too restrictive
    # weights = T / (occ_per_voxels_crossed**2)
    
    ## linear -- p=1 at min, p=0 at max (82000/120000, CV = 0.78 when a = min)
    # b, a = occ_per_voxels_crossed.max(), occ_per_voxels_crossed.min()
    # b, a = occ_per_voxels_crossed.max(), occ_per_voxels_crossed.mean()
    b, a = occ_per_voxels_crossed.max(), np.median(occ_per_voxels_crossed)
    # weights =  (b - occ_per_voxels_crossed) / (b - a)

    ## polynomial --    p=1 at min, p=0 at max -- seems too lenient
    # weights =  1 - ((occ_per_voxels_crossed - a) / (b - a))**3

    ## alt polynomial
    weights =  ((b - occ_per_voxels_crossed) / (b - a))**4

    subsampled_entries = []
    for entry, weight in zip(selected_entries, weights):
        rng = np.random.default_rng()
        p = rng.uniform(0,1)
        if p < weight:
            subsampled_entries.append(entry)

    sub_occupancy = np.zeros((ndiv, ndiv, ndiv), dtype=np.int32)       # [ix,iy,iz]
    sub_total_path = np.zeros((ndiv, ndiv, ndiv), dtype=np.float64)    # [ix,iy,iz]
    sub_dirs = []
    sub_track_lengths = []

    for entry in subsampled_entries:
        lengths, p0, p1 = crossing_info_list[entry]
        for idx, dl in lengths.items():
            # ix, iy, iz are the "coordinates" specifying a specific voxel in xyz space. args.index_order specifies the scheme by which the index idx in the flat voxel array of the voxel at (ix, iy, iz) is computed.   
            ix, iy, iz = inv_flat_index(idx, ndiv, args.index_order)
            sub_occupancy[ix, iy, iz] += 1
            sub_total_path[ix, iy, iz] += dl
        
        v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            sub_dirs.append(v / norm)
            sub_track_lengths.append(sum(lengths.values()))
        
    sub_dirs_arr = np.asarray(sub_dirs, dtype=float) if sub_dirs else np.zeros((0, 3), dtype=float)
    sub_track_lengths_arr = np.asarray(sub_track_lengths, dtype=float) if sub_track_lengths else np.zeros(0, dtype=float)
        
    make_projection_plots(args, outdir / "resampled", sub_occupancy, sub_total_path)
    make_track_direction_plots(args, outdir / "resampled", sub_dirs_arr, sub_track_lengths_arr)
    if args.make_slice_plots:
        make_slice_plots(args, outdir / "resampled", sub_occupancy, sub_total_path)

    with (outdir / "resampled" / "subselected_entries.txt").open("w") as f:
        for entry in subsampled_entries:
            f.write(f"{entry}\n")

    sub_crossed = sub_occupancy > 0
    print(f"Tracks in sub-sample: {len(subsampled_entries)}")
    print(f"Nonzero occupancy voxels in subsample: {np.count_nonzero(sub_crossed)} / {ndiv**3}")
    print(f"Median occupancy among crossed voxels: {np.median(sub_occupancy[sub_crossed]) if np.any(sub_crossed) else 0:.6g}")
    print(f"Max occupancy: {sub_occupancy.max()}")
    print(f"Standard deviation / Mean of occupancies: {sub_occupancy.std() / sub_occupancy.mean()}")
    print(f"Total path length in volume: {sub_total_path.sum():.6g} cm")
    
####################

def compute_occupancy(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ndiv = args.ndiv
    xbounds, ybounds, zbounds = build_bounds(args, ndiv)

    # tchain, files = setup_tchain(args)

    # CRT_EntryPoint = array("f", [0.0, 0.0, 0.0])
    # CRT_ExitPoint = array("f", [0.0, 0.0, 0.0])
    # tchain.SetBranchAddress("CRT_EntryPoint", CRT_EntryPoint)
    # tchain.SetBranchAddress("CRT_ExitPoint", CRT_ExitPoint)

    track_info_file = tab.open_file(args.track_info_file, mode="r")
    trajectory_table = track_info_file.root.output

    # selected_entries = select_entries(tchain, args, CRT_EntryPoint, CRT_ExitPoint, xbounds, ybounds, zbounds)
    selected_entries = select_entries(trajectory_table, args, xbounds, ybounds, zbounds)

    occupancy = np.zeros((ndiv, ndiv, ndiv), dtype=np.int32)       # [ix,iy,iz]
    total_path = np.zeros((ndiv, ndiv, ndiv), dtype=np.float64)    # [ix,iy,iz]
    dirs = []
    track_lengths = []

    print("Accumulating voxel occupancy and path length...")
    for row, entry in enumerate(selected_entries):
        if args.progress_every > 0 and row > 0 and row % args.progress_every == 0:
            print(f"  processed {row} / {len(selected_entries)} selected tracks")

        # tchain.GetEntry(int(entry))
        track = trajectory_table[entry]
        # the projected points are the equivalent of the "CRT" entry/exit points
        entry_point, exit_point = project_segment_to_detector_bounds(*get_trajectory_start_end(track)) # from select_entries, we know that track will pass the tests in get_trajectory_start_end. So it's kind of redundant to have them be done again, but hopefully it does not add too much runtime.  
        # p0 = (float(CRT_EntryPoint[0]), float(CRT_EntryPoint[1]), float(CRT_EntryPoint[2]))
        # p1 = (float(CRT_ExitPoint[0]), float(CRT_ExitPoint[1]), float(CRT_ExitPoint[2]))
        p0 = (float(entry_point[0]), float(entry_point[1]), float(entry_point[2]))
        p1 = (float(exit_point[0]), float(exit_point[1]), float(exit_point[2]))

        lengths = segment_voxel_lengths(p0, p1, xbounds, ybounds, zbounds, ndiv, args.index_order)
        if not lengths:
            continue

        for idx, dl in lengths.items():
            # ix, iy, iz are the "coordinates" specifying a specific voxel in xyz space. args.index_order specifies the scheme by which the index idx in the flat voxel array of the voxel at (ix, iy, iz) is computed.   
            ix, iy, iz = inv_flat_index(idx, ndiv, args.index_order)
            occupancy[ix, iy, iz] += 1
            total_path[ix, iy, iz] += dl

        v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            dirs.append(v / norm)
            track_lengths.append(sum(lengths.values()))

    dirs_arr = np.asarray(dirs, dtype=float) if dirs else np.zeros((0, 3), dtype=float)
    track_lengths_arr = np.asarray(track_lengths, dtype=float) if track_lengths else np.zeros(0, dtype=float)

    # write_outputs(args, outdir, occupancy, total_path, selected_entries, files, dirs_arr, track_lengths_arr)
    write_outputs(args, outdir, occupancy, total_path, selected_entries, dirs_arr, track_lengths_arr)
    make_projection_plots(args, outdir, occupancy, total_path)
    make_slice_summary_plots(args, outdir, occupancy, total_path)
    make_track_direction_plots(args, outdir, dirs_arr, track_lengths_arr)
    if args.make_slice_plots:
        make_slice_plots(args, outdir, occupancy, total_path)

    if args.resample:
        resample_balanced_occupancy(args, outdir, trajectory_table, selected_entries, occupancy)

    crossed = occupancy > 0
    print("\nDone")
    print("----")
    print(f"Selected tracks: {len(selected_entries)}")
    print(f"Nonzero occupancy voxels: {np.count_nonzero(crossed)} / {ndiv**3}")
    print(f"Median occupancy among crossed voxels: {np.median(occupancy[crossed]) if np.any(crossed) else 0:.6g}")
    print(f"Max occupancy: {occupancy.max()}")
    print(f"Total path length in volume: {total_path.sum():.6g} cm")
    print(f"Outputs written to: {outdir}")


def main() -> None:
    args = parse_args()
    compute_occupancy(args)


if __name__ == "__main__":
    main()
