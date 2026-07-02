# Physics-Prior Few-Sample Grid Localization

This repository contains a Python prototype for **Physics-Prior Few-Sample Grid Localization**.

The project focuses only on generic 2D computer vision: locating fixed 10x10 or 15x15 grid patterns in camera images with limited samples. It combines geometric priors, classical image processing, robust lattice fitting, local center refinement, ROI statistics, and localization-error evaluation.

No application-specific background is required to use or review this repository.

## Main Files

- `pg_grid.py`: core localization pipeline.
- `pg_grid_eval.py`: annotation loading, prediction loading, localization metrics, and report generation.
- `pg_grid_demo.py`: command-line single-image pipeline runner.
- `pg_grid_annotator.py`: semi-automatic point annotation initializer and OpenCV-based correction tool.
- `pg_grid_evaluate.py`: command-line localization-evaluation runner.
- `tests/`: pytest tests based on synthetic grid images and synthetic point records.

## Algorithm Summary

The current implementation follows this pipeline:

1. Locate the main rectangular target region.
2. Rectify the region into a canonical square view.
3. Generate or fit a physically constrained grid.
4. For 10x10 patterns, enhance dark square candidates and fit a regular lattice.
5. Refine each local center with component-level constraints.
6. Extract ROI statistics and write CSV/JSON outputs.
7. Compare predicted grid points with annotation points and generate localization reports.

## Quick Start

```powershell
pip install -r requirements.txt
python -m pytest tests -v
```

The repository includes two neutral synthetic sample images:

- `examples/synthetic_10x10_dark_squares.png`
- `examples/synthetic_15x15_bright_points.png`

You can regenerate them with:

```powershell
python examples/generate_synthetic_samples.py
```

Run the included 10x10 sample:

```powershell
python pg_grid_demo.py --image examples/synthetic_10x10_dark_squares.png --grid 10 --output outputs/synthetic_10x10
```

Run the included 15x15 sample:

```powershell
python pg_grid_demo.py --image examples/synthetic_15x15_bright_points.png --grid 15 --output outputs/synthetic_15x15
```

Run another image:

```powershell
python pg_grid_demo.py --image "path/to/image.jpg" --grid 10 --output outputs/sample_10x10
```

Create initial annotation from prediction:

```powershell
python pg_grid_annotator.py --image outputs/sample_10x10/rectified_chip.jpg --prediction outputs/sample_10x10/result.json --grid 10 --output annotations/sample_10x10_initial.json --init-only
```

Evaluate localization error:

```powershell
python pg_grid_evaluate.py --annotation annotations/sample_10x10_initial.json --prediction outputs/sample_10x10/result.json --image outputs/sample_10x10/rectified_chip.jpg --output reports/sample_10x10_initial
```

## Outputs

Each pipeline run writes:

- `overlay_grid.jpg`
- `rectified_chip.jpg`
- `roi_debug.jpg`
- `values.csv`
- `result.json`, including `grid_points`

Each evaluation run writes:

- `localization_metrics.json`
- `localization_errors.csv`
- `localization_error_overlay.jpg`

## Notes For Further Optimization

Useful next steps include robust lattice fitting, row/column consistency correction, local template matching, and better outlier rejection. Changes should remain test-driven and should preserve existing output file names and JSON fields.
