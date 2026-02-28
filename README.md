# 3D Structure-from-Motion (Photogrammetry) Image Creation

Open-source MSc final project on cultural heritage digitization using both classical SfM and neural reconstruction workflows.

## Overview

This repository packages the custom code and configuration used in the thesis:

- Traditional photogrammetry support steps (image conversion, resizing, feature checks)
- Neural pipelines based on `NeRO` and `NeuS`
- Custom experiment configs for heritage-object datasets
- Sample qualitative comparison outputs

Title used in the dissertation:
`3D Structure-from-Motion (Photogrammetry) Image Creation: An Open-Source Solution for Cultural Heritage Digitization`

## Repository Structure

```text
scripts/
  preprocess/
  analysis/
pipelines/
  nero/
  neus/
configs/
  nero_custom/
assets/
  comparison_images/
```

## Included Custom Components

### Preprocessing scripts

- `scripts/preprocess/convert_arw_to_png.py`
- `scripts/preprocess/convert_tif_to_png.py`
- `scripts/preprocess/reduce_size.py`

### Feature analysis scripts

- `scripts/analysis/feature_extraction.py`
- `scripts/analysis/feature_matching.py`

### Neural pipeline scripts

- `pipelines/nero/nero_pipeline.py`
- `pipelines/nero/extract_colored_mesh.py`
- `pipelines/neus/extract_colored_mesh.py`

### Custom NeRO configs

- `configs/nero_custom/*.yaml`

### Sample output comparisons

- `assets/comparison_images/*`

## Notes on Reproducibility

- The scripts were developed in a research environment and some contain dataset-specific absolute paths that should be updated before reuse.
- The NeRO pipeline script expects external dependencies and environments (Conda, COLMAP, CUDA, NeRO codebase dependencies).
- This repository intentionally excludes large raw datasets and training artifacts.

## External Dependencies (upstream projects)

- COLMAP: https://github.com/colmap/colmap
- NeRO: https://github.com/liuyuan-pal/NeRO
- NeuS: https://github.com/Totoro97/NeuS
- NeRF baseline reference: https://github.com/bmild/nerf

## License and Attribution

This repository contains custom project code/configuration plus references to external frameworks.
Please respect and retain upstream licenses when reproducing full pipelines.

