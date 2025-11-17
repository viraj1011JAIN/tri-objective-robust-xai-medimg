# Raw data layout (external vault)

To keep the repository lightweight, raw medical imaging datasets are stored
*outside* the git repo on a dedicated data drive.

On the development machine:

- `DATA_ROOT = F:\data`

with the following subdirectories:

- `${DATA_ROOT}/derm7pt`
- `${DATA_ROOT}/isic_2018`
- `${DATA_ROOT}/isic_2019`
- `${DATA_ROOT}/isic_2020`
- `${DATA_ROOT}/nih_cxr`
- `${DATA_ROOT}/padchest`

All dataset configuration files under `configs/datasets/*.yaml` refer to these
paths using `${DATA_ROOT}/...`.

Preprocessed artefacts (resized images, tensors, cached splits, etc.) are
written *inside* this repository under:

- `data/processed/...`

This separation keeps the repo small while making the raw-data layout explicit
and reproducible.
