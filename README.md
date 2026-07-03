# Physics-Prior Few-Sample Grid Localization

This repository contains a Python prototype for **Physics-Prior Few-Sample Grid Localization**.

The project focuses only on generic 2D computer vision: locating fixed 10x10 or 15x15 grid patterns in camera images with limited samples. It combines geometric priors, classical image processing, robust lattice fitting, local center refinement, ROI statistics, and localization-error evaluation.

No application-specific background is required to use or review this repository.

## Main Files

- `pg_grid.py`: core localization pipeline.
- `pg_quant.py`: PG-Quant V1.0 unit-level quantification (ROI + background annulus, self-referenced flat-field correction, color and intensity feature sets, per-unit quality flags).
- `pg_quant_viz.py`: unit-level visualizations (intensity/SNR heatmaps, corrected-color map, reliability overlay).
- `pg_benchmark.py`: seeded synthetic perturbation benchmark (7 perturbation families) with degradation-curve reports and A/B comparison.
- `pg_benchmark_demo.py`: command-line benchmark runner and report comparator.
- `pg_grid_eval.py`: annotation loading, prediction loading, localization metrics, and report generation.
- `pg_grid_demo.py`: command-line single-image pipeline runner.
- `pg_quant_demo.py`: command-line quantification runner based on an existing `result.json`.
- `pg_grid_annotator.py`: semi-automatic point annotation initializer and OpenCV-based correction tool.
- `pg_grid_evaluate.py`: command-line localization-evaluation runner.
- `docs/next_stage_design.md`: V2 design document (localization + quantification roadmap).
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
5. Lattice bundle adjustment (V2.0): local refinement collects observations
   with real image evidence, an 8-DoF homography over `(row, col) -> (x, y)`
   is fitted with Tukey-IRLS (the projective terms absorb residual perspective
   that an affine model cannot express), windows are re-centered on the model
   prediction for a second refinement pass, and the final fit assigns each
   point a `confidence`, a `source` (`candidate_refined` vs `model_imputed`),
   and `flags`. Only evidence-backed observations vote, which mechanically
   removes the "regular-but-wrong majority" failure mode, and a candidate
   support ratio provides an external-evidence check: grids whose points are
   not backed by detected blobs are marked untrusted and downgrade the frame
   quality with machine-readable `reasons`. Diagnostics are written to
   `result.json` as `lattice_consistency` (legacy keys preserved).
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

## PG-Quant V1.0: Unit-Level Quantification

After localization, the pipeline quantifies every array unit on the rectified
image:

1. **Circular signal ROI + background annulus**: each unit gets a conservative
   disk ROI (0.18 pitch) and a surrounding background ring (0.30–0.44 pitch)
   sampled from the gap between units, providing a local background reference.
2. **Self-referenced flat-field correction**: the annulus medians of all units
   are themselves samples of the illumination field. A second-order polynomial
   is fitted to them (with robust re-fitting and a constant-field fallback) and
   used for multiplicative correction — no external calibration target needed.
3. **Two feature sets**: color features (raw and corrected BGR medians,
   chromaticity, Lab) and intensity features (signed background-subtracted
   signal, relative signal ratio, field-corrected signal, integrated signal,
   SNR against annulus noise).
4. **Per-unit quality flags**: `saturated`, `under_exposed`, `low_snr`,
   `roi_out_of_bounds`, `non_uniform` (ROI contamination fraction that would
   threaten the median), `background_anomaly` (annulus deviating from the
   fitted illumination field), summarized into a `quant_reliable` boolean.

Re-run quantification on an existing localization result:

```powershell
python pg_quant_demo.py --result outputs/sample_10x10/result.json
```

## Quantification Visualizations

Every pipeline run also renders four unit-level views:

- `quant_overlay.jpg`: rectified image with ROI circles colored by
  `quant_reliable` (green/red), background annulus rings, and flag codes next
  to unreliable units (S=saturated, U=under-exposed, N=low SNR, B=out of
  bounds, H=non-uniform, A=background anomaly, I=imputed).
- `quant_heatmap_intensity.jpg`: grid heatmap of the corrected
  background-subtracted signal (Viridis colormap with a value scale bar;
  unreliable cells are crossed out).
- `quant_heatmap_snr.jpg`: grid heatmap of per-unit SNR.
- `quant_color_map.jpg`: each cell filled with the unit's corrected BGR color —
  the most direct view of the per-unit color distribution.

## Perturbation Benchmark and A/B Comparison

`pg_benchmark.py` generates fully seeded synthetic scenes (known unit centers
and a known photometric gradient across units) and sweeps seven perturbation
families — rotation, perspective, blur, illumination gradient, occlusion,
glare, and noise — each across an intensity ladder. Every case runs the full
pipeline and records localization error (px and % of pitch), quantification
rank-order fidelity (Spearman against the known gradient), per-unit
reliability ratio, and geometry-trust telemetry, producing degradation-curve
data rather than single-point metrics.

```powershell
python pg_benchmark_demo.py --grid 10 --seeds 2 --output reports/bench_v2
python pg_benchmark_demo.py --compare reports/bench_a/benchmark_report.json reports/bench_b/benchmark_report.json
```

Because scenes are byte-reproducible for a given seed, comparing two code
versions only requires running the same benchmark on each checkout and diffing
the reports; the comparator aligns cases, prints per-family metric deltas, and
lists regressed cases. Images are generated on demand and never committed.

## Outputs

Each pipeline run writes:

- `overlay_grid.jpg`
- `rectified_chip.jpg`
- `roi_debug.jpg`
- `values.csv`
- `result.json`, including `grid_points` (each point now carries
  `confidence`/`source`/`flags`), `lattice_consistency`, and `quant_summary`
- `quant_values.csv`: per-unit features and quality flags
- `quant_result.json`: quantification metadata, illumination model, and per-unit records

Each evaluation run writes:

- `localization_metrics.json`
- `localization_errors.csv`
- `localization_error_overlay.jpg`

## Notes For Further Optimization

Useful next steps include robust lattice fitting, row/column consistency correction, local template matching, and better outlier rejection. Changes should remain test-driven and should preserve existing output file names and JSON fields.
