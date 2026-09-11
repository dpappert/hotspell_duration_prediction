# CNN gridded (2D) input data

This folder is a **placeholder**. The gridded input arrays used by the CNN
model (3x3 grid, per-member, per-region) are ~2GB per region (~6GB total
across the 3 regions) and are not shipped in this repository.

Expected contents once populated:

```
data/cnn_grid/
├── cl1/r*.nc      # region 1 (50 member files)
├── cl2/r*.nc      # region 2 (50 member files)
└── cl4/r*.nc      # region 4 (50 member files)
```

Each `.nc` file is a per-ensemble-member NetCDF array read via `xarray` and
concatenated along the `event` dimension (see `src/hwml/data.py`).

**To get this data:** [contact details / link to external hosting — fill in
before publishing]. Until this data is in place, any script that trains or
evaluates the CNN model will fail with a clear `FileNotFoundError` pointing
here — every other model (RF, GAM, LR, MLP) runs fine without it.
