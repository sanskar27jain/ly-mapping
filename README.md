# LY Mapping

Tools for reconstructing and visualizing three-dimensional light-yield maps using crossing-muon data.

The reconstruction treats the detector volume as a regular three-dimensional voxel grid. For each crossing-muon track, the path length through every intersected voxel is calculated and used to construct a sparse linear system,

[
A x = b,
]

where:

* (A) contains the energy deposited by each track in each voxel,
* (x) contains the unknown voxel light yields in PE/MeV,
* (b) contains the detected-photon count associated with each track.

The solution is constrained to be nonnegative and may include nearest-neighbor L2 smoothness regularization.

## Repository contents

### `pds_cal_gen_scipy_sparse_l2smooth.py`

Python reconstruction script that:

* reads crossing-muon trajectories and detected-photon records from ROOT files,
* associates detected photons with each trajectory,
* calculates track intersections with the detector voxel grid,
* constructs the sparse response matrix (A) and observation vector (b),
* optionally applies nearest-neighbor L2 smoothness regularization,
* solves the nonnegative least-squares problem with SciPy,
* writes the reconstructed voxel light yields to a text file.

The included data-loading code is configured as an example for the Angelo2 simulation files. Users will generally need to modify the ROOT tree names, branch names, optical-channel selection, file naming pattern, detector geometry, or event association for their own dataset.

### `liang_barsky.C`

C++ implementation of the three-dimensional Liang–Barsky line-clipping algorithm.

The Python solver loads this file through PyROOT and uses it to calculate the entry and exit points of each muon track within every voxel. This file must be available when running the solver.

### `pds_cal_plot.C`

ROOT macro for reading a reconstructed solution and producing:

* summed YX, ZX, and YZ projections,
* average YX, ZX, and YZ projections,
* individual two-dimensional slices along each detector axis.

The macro reads the voxel count from the solution file and fills a ROOT `TH3D` histogram using the expected flattened voxel ordering.

## Requirements

### Python

Python 3.9 or newer is recommended.

The reconstruction script requires:

* NumPy
* SciPy
* PyROOT

Install the Python-only dependencies with:

```bash
python3 -m pip install numpy scipy
```

PyROOT is distributed as part of ROOT and normally should not be installed independently with `pip`.

### ROOT

ROOT 6.26 or newer is recommended.

ROOT must include Python bindings compatible with the Python environment used to run the solver. Verify the installation with:

```bash
root-config --version
python3 -c "import ROOT; print(ROOT.gROOT.GetVersion())"
```

Both commands should report the same ROOT installation.

### External dependencies

The solver requires:

* ROOT input files containing the crossing-muon and detected-photon trees,
* the `liang_barsky.C` source file,
* sufficient memory and storage for the selected event count and voxel resolution.

No large input datasets or generated reconstruction files are included in this repository.

## Input data assumptions

The example solver is configured for Angelo2 ROOT files with names of the form:

```text
run0_rsl100_abs20_500evts.root
run1_rsl100_abs20_500evts.root
...
```

The default implementation expects two ROOT trees:

```text
TrajCut/CRTTrajectoryCut
XAresponse/DetectedPhotons
```

The trajectory tree is expected to contain:

```text
CRT_EntryPoint
CRT_ExitPoint
Event
```

The detected-photon tree is expected to contain:

```text
EventID
OpChannel
```

The script associates the two trees using the ROOT chain file number and event identifier.

Users working with another data format should modify the corresponding file-list, tree, branch, channel-selection, and event-matching sections.

## Detector geometry

The reconstruction divides a rectangular detector volume into `ndiv` equal divisions along each axis.

The default example geometry is:

```text
x: -400 to 400 cm
y: -425 to 425 cm
z: -300 to 600 cm
```

These bounds must correspond to the physical region represented by the trajectories and by the requested solution.

### Critical plotting requirement

The values of `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, and `zmax` in `pds_cal_plot.C` must exactly match the bounds used by the solver when the solution was generated.

The solution text file records the voxel count and voxel values, but it does not record the physical coordinate bounds. Using different plotting bounds will assign the reconstructed values to incorrect physical positions and produce geometrically incorrect projections and slices.

## Voxel ordering

The solution is flattened into a one-dimensional array. The solver and plotting code must use the same voxel ordering.

The plotting macro assumes:

```cpp
index = ix + iy * ndiv + iz * ndiv * ndiv;
```

Therefore:

1. x changes fastest,
2. y changes next,
3. z changes slowest.

Changing the loop ordering in either script requires updating the indexing convention in the other script.

## Reconstruction model

For each track and voxel intersection, the matrix element is calculated as:

$$
A_{ij} = \ell_{ij}\left(\frac{dE}{dx}\right),
$$

where:

- $\ell_{ij}$ is the length of track $i$ inside voxel $j$,
- $dE/dx$ is the assumed muon energy loss.

The example code uses:

$$
\frac{dE}{dx} = 2.1\ \mathrm{MeV/cm}.
$$

The reconstructed values therefore have units of PE/MeV when the observation
vector contains detected photoelectrons, or detected-photon counts interpreted
as photoelectrons.

## Smoothness regularization

The optional smoothness term penalizes differences between neighboring voxels:

$$
\frac{\lambda}{2}
\sum_{\langle j,k\rangle}
\left(x_j-x_k\right)^2.
$$

The resulting problem is:

$$
\min_{x \geq 0}
\left[
\frac{1}{2}\lVert Ax-b\rVert_2^2
+
\frac{\lambda}{2}
\sum_{\langle j,k\rangle}
\left(x_j-x_k\right)^2
\right].
$$

Each voxel is connected to its immediate neighbors along the x, y, and z
directions. Each neighboring pair is counted once.

Set `--lambda-smooth 0` to disable regularization.

The appropriate value of (lambda) depends on the dataset, voxel resolution, event count, detector geometry, and scaling of the linear system. It should generally be selected through a scan using a suitable validation metric rather than assumed to be universal.

## Running the solver

A typical reconstruction command is:

```bash
python3 pds_cal_gen_scipy_sparse_l2smooth.py \
    --data-dir path/to/data/ \
    --ndiv 12 \
    --num-events 60000 \
    --lambda-smooth 3e6 \
    --outname solutions/test_12x12x12_60000evts_lambda3e6.txt
```

The main arguments are:

* `--data-dir`: directory containing the Angelo2 ROOT files,
* `--ndiv`: number of voxel divisions along each axis,
* `--num-events`: maximum number of trajectory entries to process,
* `--lambda-smooth`: nearest-neighbor smoothness strength,
* `--outname`: output solution filename.

For an unregularized solution:

```bash
python3 pds_cal_gen_scipy_sparse_l2smooth.py \
    --data-dir path/to/data/ \
    --ndiv 12 \
    --num-events 60000 \
    --lambda-smooth 0 \
    --outname solutions/test_12x12x12_60000evts_unregularized.txt
```

When `liang_barsky.C` is not located beside the Python script, provide its path explicitly:

```bash
python3 pds_cal_gen_scipy_sparse_l2smooth.py \
    --data-dir path/to/data/ \
    --liang-barsky /full/path/to/liang_barsky.C \
    --ndiv 12 \
    --num-events 60000 \
    --lambda-smooth 3e6 \
    --outname solutions/test_12x12x12_60000evts_lambda3e6.txt
```

Create the output directory before running if necessary:

```bash
mkdir -p solutions
```

## Solution-file format

The solver writes a plain-text solution file.

The first line contains:

```text
ndiv
```

The following `ndiv^3` lines contain one reconstructed voxel value per line:

```text
x[0]
x[1]
x[2]
...
```

For example, a `12 × 12 × 12` reconstruction contains:

```text
1 + 12^3 = 1729
```

total lines.

## Plotting the solution

Before running the plotting macro, edit `pds_cal_plot.C` and set:

1. the solution-file path,
2. `xmin` and `xmax`,
3. `ymin` and `ymax`,
4. `zmin` and `zmax`,
5. the output directory and output filenames, if desired.

Ensure the output directory exists:

```bash
mkdir -p test_plot
```

Run the macro with ROOT:

```bash
root -l -b -q pds_cal_plot.C
```

The macro produces summed and average projections in the three coordinate planes, along with individual voxel slices.

The average projections are obtained by dividing each summed projection by `ndiv`, corresponding to the number of voxel layers included in the projection.

## Data and generated files

Large input ROOT files are not stored in this repository.

Generated products are also excluded, including:

* reconstructed solution text files,
* sparse matrices,
* observation vectors,
* diagnostic logs,
* PNG figures,
* PDF figures,
* ROOT output files.

A typical local directory structure is:

```text
LY_mapping/
├── pds_cal_gen_scipy_sparse_l2smooth.py
├── liang_barsky.C
├── pds_cal_plot.C
├── data/
│   └── specific_dataset/
├── solutions/
└── test_plot/
```

The `data`, `solutions`, and `test_plot` directories may be excluded from version control except for optional placeholder files such as `.gitkeep`.

## Adapting the code

This repository is intended to provide a minimal example of the reconstruction procedure rather than a detector-independent framework.

For another dataset, users may need to change:

* ROOT file discovery,
* tree and branch names,
* trajectory coordinate definitions,
* event matching,
* photon-count construction,
* optical-channel selection,
* detector coordinate bounds,
* energy-loss assumptions,
* event-selection criteria,
* voxel indexing,
* output naming,
* regularization selection.

Any changes to detector bounds or voxel ordering must be applied consistently in both the reconstruction and plotting scripts.

## Performance considerations

Runtime is primarily determined by:

* the number of detected-photon records,
* the number of trajectories,
* the number of voxels,
* the number of track–voxel intersection tests,
* the convergence behavior of the bounded least-squares solver.

The example Angelo2 dataset may contain billions of detected-photon records. Reading and aggregating this tree can therefore take substantially longer than processing the trajectory tree.

Increasing `ndiv` increases the number of unknowns as:

$$
[
N_{\mathrm{voxels}} = \mathrm{ndiv}^3.
]
$$

The current example tests every accepted trajectory against every voxel. Consequently, the matrix-construction cost grows rapidly with both event count and voxel resolution.
