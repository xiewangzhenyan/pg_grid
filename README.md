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

1. Locate the main rectangular target region with a two-tier detector: the
   first tier keeps the original high-percentile threshold and small-target
   area prior, and a second tier (Otsu threshold plus a relaxed area limit)
   only activates when the first tier finds no candidate, so large panels
   that fill nearly half of the frame are still detected.
2. Rectify the region into a canonical square view.
3. Fit a physically constrained grid from blob candidates: black-hat enhanced
   dark squares for 10x10, top-hat enhanced bright dots for 15x15. A small
   angle sweep first estimates the residual lattice rotation left over after
   rectification, the candidates are de-rotated, row/column clusters are
   matched against a regular equally spaced axis model, and the resulting grid
   is rotated back. Projection-peak fitting and the uniform physical grid stay
   as fallbacks.
4. Refine each local center with component-level constraints.
5. Enforce global lattice consistency: fit a robust affine lattice model over
   `(row, col) -> (x, y)` with iterative thresholded re-fitting, then snap
   points pulled away by thin dark lines, local shadows, or highlights back to
   the model prediction. Inliers keep their locally refined positions, and the
   correction diagnostics are written to `result.json` as `lattice_consistency`.
6. Extract ROI statistics and write CSV/JSON outputs.
7. Compare predicted grid points with annotation points and generate localization reports.

Robustness notes:

- Axis-cluster selection pre-filters clusters by support before enumerating
  combinations, which bounds the search to roughly `C(count + 4, 4)` and
  avoids the combinatorial blow-up caused by spurious clusters from screws or
  edge fragments.
- Projection-peak selection collects more peaks than strictly needed and picks
  the most regular equally spaced subset, so bright halo overshoot near panel
  edges cannot displace true rows or columns.
- The bright-dot detector estimates its Otsu threshold from the image interior
  only, because the step transition at panel borders produces a top-hat band
  far stronger than the dots themselves.
- The lattice-consistency step skips grids smaller than 3x3 and refuses to
  correct anything when fewer than half of the points agree with the fitted
  model, so a globally wrong fit can never drag valid points away.

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
