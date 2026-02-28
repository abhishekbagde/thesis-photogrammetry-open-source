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

## Visual Comparison

Methods shown below include:
`Agisoft Metashape`, `COLMAP`, `NeRF`, `NeRO`, `NeuS`, and `VisualSFM`.

### Wesley

| Agisoft Metashape | COLMAP | NeRF |
|---|---|---|
| ![Wesley Agisoft](assets/comparison_images/wesley_agisoft_metashape.png) | ![Wesley COLMAP](assets/comparison_images/wesley_colmap.png) | ![Wesley NeRF](assets/comparison_images/wesley_nerf.png) |

| NeRO | NeuS | VisualSFM |
|---|---|---|
| ![Wesley NeRO](assets/comparison_images/wesley_nero.png) | ![Wesley NeuS](assets/comparison_images/wesley_neus.png) | ![Wesley VisualSFM](assets/comparison_images/wesley_visualsfm.png) |

### Cuneiform

| Agisoft Metashape | COLMAP | NeRF |
|---|---|---|
| ![Cuneiform Agisoft](assets/comparison_images/cuneiform_agisoft_metashape.png) | ![Cuneiform COLMAP](assets/comparison_images/cuneiform_colmap.png) | ![Cuneiform NeRF](assets/comparison_images/cuneiform_nerf.png) |

| NeRO | NeuS | VisualSFM |
|---|---|---|
| ![Cuneiform NeRO](assets/comparison_images/cuneiform_nero.png) | ![Cuneiform NeuS](assets/comparison_images/cuneiform_neus.png) | ![Cuneiform VisualSFM](assets/comparison_images/cuneiform_visualsfm.png) |

### Cuneiform930

| Agisoft Metashape | COLMAP | NeRF |
|---|---|---|
| ![Cuneiform930 Agisoft](assets/comparison_images/cuneiform930_agisoft_metashape.png) | ![Cuneiform930 COLMAP](assets/comparison_images/cuneiform930_colmap.png) | ![Cuneiform930 NeRF](assets/comparison_images/cuneiform930_nerf.png) |

| NeRO | NeuS | VisualSFM |
|---|---|---|
| ![Cuneiform930 NeRO](assets/comparison_images/cuneiform930_nero.png) | ![Cuneiform930 NeuS](assets/comparison_images/cuneiform930_neus.png) | ![Cuneiform930 VisualSFM](assets/comparison_images/cuneiform930_visualsfm.png) |

### Gaster Amulet 49D

| Agisoft Metashape | COLMAP | NeRF |
|---|---|---|
| ![Gaster49D Agisoft](assets/comparison_images/gasteramulet49d_agisoft_metashape.png) | ![Gaster49D COLMAP](assets/comparison_images/gasteramulet49d_colmap.png) | ![Gaster49D NeRF](assets/comparison_images/gasteramulet49d_nerf.png) |

| NeRO | NeuS | VisualSFM |
|---|---|---|
| ![Gaster49D NeRO](assets/comparison_images/gasteramulet49d_nero.png) | ![Gaster49D NeuS](assets/comparison_images/gasteramulet49d_neus.png) | ![Gaster49D VisualSFM](assets/comparison_images/gasteramulet49d_visualsfm.png) |

Note: `.ply` mesh outputs are present in the original thesis workspace but are not embedded in README since GitHub does not render `.ply` previews inline.

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
