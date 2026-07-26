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
- `pg_fluoro_sim.py`: physically-modelled fluorescence array simulator with ground truth — generates emission images (wavelength→RGB, per-well intensities, optics, sensor noise) and evaluates the pipeline against the known truth. `--concentration` drives emission from **absolute fluorophore concentration** through an inner-filter/self-quenching model, and `--dataset N` pools the results into a calibration dataset.
- `pg_colori_sim.py`: transmission (colorimetric) array simulator with ground
  truth — white EL backlight through a black mask, per-wavelength Beer-Lambert
  absorption, paired blank image, and `A = -log10(I/I_blank)` evaluation.
  Shares the geometry/optics/sensor chain with `pg_fluoro_sim`.
- `pg_unit_export.py`: per-unit tight cropping — exports every array unit as an
  individual background-free PNG with a manifest and a contact-sheet
  verification image.
- `pg_grid_eval.py`: annotation loading, prediction loading, localization metrics, and report generation.
- `pg_grid_demo.py`: command-line single-image pipeline runner.
- `pg_quant_demo.py`: command-line quantification runner based on an existing `result.json`.
- `pg_grid_annotator.py`: semi-automatic point annotation initializer and OpenCV-based correction tool.
- `pg_grid_evaluate.py`: command-line localization-evaluation runner.
- `docs/next_stage_design.md`: V2 design document (localization + quantification roadmap).
- `tests/`: pytest tests based on synthetic grid images and synthetic point records.

## Algorithm Summary

The current implementation follows this pipeline:

0. Detect unit polarity automatically: whether units are darker or brighter
   than the panel depends on the imaging mode, not the grid size. Both dark
   (black-hat) and bright (top-hat) candidates are extracted; lattice-fit
   success adjudicates, ordered by candidate-count closeness and an
   interior mean-vs-median statistical hint. The projection fallback uses the
   statistical hint only — the bright gaps of a dark-unit lattice form a
   complementary regular lattice, so blindly trying the opposite polarity
   would yield a confident half-pitch-shifted grid. The decision is recorded
   in `result.json` as `unit_polarity`.
1. Locate the main rectangular target region with an **area-fraction-agnostic**
   detector. Candidates are generated at two thresholds — a high percentile
   (historical behavior, 20% area cap) and Otsu (92% area cap) — and *all* of
   them are scored together rather than returning as soon as the first tier
   yields something. Scoring combines area, brightness, and **rectangular fill
   ratio** (contour area over its min-area-rect area), a purely shape-based
   criterion that is independent of how much of the frame the target occupies.
   This matters because a fixed high-percentile threshold implicitly assumes
   the target covers only a small fraction of the image: crop the background
   away and the threshold is forced upward until the mask cuts *inside* the
   panel instead of at its border. Otsu maximizes inter-class variance and
   presumes no area fraction, so it stays correct whether the target covers
   12% or 83% of the frame, and the fill ratio distinguishes a complete panel
   contour (≈0.9–1.0) from a fragment carved out of the panel interior (≈0.5).
2. Rectify the region into a canonical square view.
3. Fit a physically constrained grid from blob candidates: black-hat enhanced
   dark units, top-hat enhanced bright units, selected by the detected
   polarity rather than by grid size. A small angle sweep first estimates the
   residual lattice rotation left over after rectification, the candidates are
   de-rotated, row/column clusters are matched against a regular equally
   spaced axis model, and the resulting grid is rotated back. Projection-peak
   fitting and the uniform physical grid stay as fallbacks.

   The detectors' geometric gates and the axis pitch window are anchored to
   the 15x15 reference and converted by unit pitch, because a unit's pixel
   size tracks `side*(1-2*margin)/(grid_size-1)` rather than the canvas side.
   Expressing them as fractions of the side silently encoded `grid_size ≈ 15`;
   at 10x10 that rejected normal units for exceeding `max_area` and left the
   axis pitch window with 3% headroom instead of 60%.

   An axis whose clusters are incomplete is completed against the regular
   lattice prior, tolerating up to half the rows or columns missing. Whole
   dark rows are the norm rather than the exception for self-luminous arrays:
   a log dilution series puts its dimmest wells below the substrate
   autofluorescence floor, leaving 6 of 10 rows detectable at 10x10 and 9 of
   15 at 15x15. When one axis resolves and the other does not, the resolved
   axis supplies its pitch as a prior — a square array has the same pitch on
   both axes, and rows going dark together leaves the column axis intact.
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

## Per-Unit Crop Export

`pg_unit_export.py` cuts every localized unit out of the rectified image as an
individual PNG that contains exactly the unit body and no background:

1. **Tight boxes, not fixed windows**: a polarity-aware local segmentation
   (Otsu inside a window smaller than half the pitch, so a neighbor can never
   be captured) yields the connected component containing the lattice point,
   and its tight bounding box is exported.
2. **Zero missed crops**: the output count always equals `grid_size²`. Units
   whose segmentation fails (occlusion, weak contrast) fall back to a
   median-sized box centered on the bundle-adjusted lattice point and are
   labeled `fallback_median_box` in the manifest.
3. **Verifiable at a glance**: `crop_manifest.json` records per-unit geometry
   and source, and `crop_contact_sheet.jpg` tiles all crops in grid order
   (green border = segmented, red = fallback).

```powershell
python pg_unit_export.py --result outputs/sample/result.json --output crops/sample
python pg_unit_export.py --image "examples/photo.jpg" --grid 15 --output crops/photo
```

On the bundled real 15x15 photo this exports 225/225 crops with zero
fallbacks; crops are lossless PNGs suitable for downstream per-unit analysis.

## Fluorescence Simulation with Ground Truth

`pg_fluoro_sim.py` renders self-luminous (fluorescent) arrays through a
physical imaging chain, so the detection limit it reports is meaningful rather
than an artifact of drawing colored squares:

1. **Scene radiance** (linear): per-well emission with a configurable intensity
   pattern (log dilution series by default, spanning from below the noise floor
   to saturation), substrate autofluorescence, filter leakage.
2. **Excitation and optics**: illumination gradient, inter-well optical
   crosstalk, defocus PSF, vignetting, lateral chromatic aberration.
3. **Geometry**: rotation, perspective, radial distortion (forward-mapped for
   ground-truth points, numerically inverted for image resampling).
4. **Sensor**: exposure to electrons, Poisson shot noise, dark current,
   Gaussian read noise, PRNU fixed-pattern noise, hot pixels, full-well
   clipping, gamma, 8-bit quantization, optional JPEG compression.

Emission wavelength maps to RGB via a CIE piecewise approximation — 530 nm
gives the expected yellow-green. Ground truth (per-well centers after all
geometric transforms, plus set intensities) is written alongside the image.

```powershell
python pg_fluoro_sim.py --grid 15 --output sim/f15 --evaluate
python pg_fluoro_sim.py --grid 15 --output sim/scan --intensity-scan
python pg_fluoro_sim.py --grid 15 --output sim/sweep --sweep
```

### Concentration ground truth

`--concentration` makes per-well **absolute concentration** the independent
variable: concentration → photophysics → intensity → imaging chain. The truth
JSON gains `concentrations` (µM), `concentration_unit`, a derived
`concentration_fraction`, the `photophysics` parameters, and the
`exposure_gain` used to convert relative emission into full-well fraction.

The model is deliberately not linear, because fluorescein is not:

```
F = Φ_eff · (1 − 10^(−A))    · 10^(−overlap·A)   ,  A = ε·c·l
    self-quench  primary IFE   secondary IFE (reabsorption)
```

The turnover position is set by A = ε·c·l, so **the liquid layer depth decides
how much of the range stays invertible**. The default is a 0.5 mm layer:

| Path length | Turnover | 1 nM – 100 µM strictly monotonic? |
|---|---|---|
| 3.0 mm | 51.5 µM | no — sensitivity already degraded past 20 µM |
| 1.0 mm | 151.5 µM | no |
| **0.5 mm (default)** | **295.3 µM** | **yes** |
| 0.2 mm | 690.1 µM | yes |

Measured at the default (ε = 8×10⁴ M⁻¹cm⁻¹, Φ = 0.92, l = 0.5 mm):

| Concentration | Emission relative to a straight line through the origin |
|---|---|
| 0.1 µM | 99.9% |
| 1 µM | 99.5% |
| 10 µM | 94.9% |
| 50 µM | 77.6% |
| 100 µM | 61.2% — compressed but still rising |

A thinner layer trades sensitivity for range: it absorbs less excitation, so
the whole curve is weaker and the detection limit rises. Measured on the
dataset command below, moving 3 mm → 0.5 mm pushed the turnover from 51 µM to
295 µM but raised the LOD from 0.29 µM to 3.13 µM. Both are the same A, seen
from two sides.

That LOD is set by **substrate autofluorescence**, not by read noise, so
longer exposure does not lower it — the background scales with exposure too.
Reaching lower concentrations needs a cleaner substrate or a better emission
filter.

Recording absolute concentration rather than "percent of maximum" is what
makes any of this expressible. Past the turnover the same grey level
corresponds to two different concentrations, and the position of that
ambiguity is fixed in µM — a relative scale would slide it around with
whatever `C_max` a consumer picked afterwards, and images with different
ranges could not share one calibration curve. `--linear` disables both
nonlinearities for an A/B control; it degenerates to the dilute-limit
expansion so the two datasets coincide at low concentration and any
difference is attributable to the nonlinearity itself.

Self-quenching is modelled but contributes under 1% below 100 µM (0.99% at
100 µM) and cannot produce a turnover on its own — the curve with only
self-quenching enabled is monotonic. The turnover is the inner filter effect.

### Generating a calibration dataset

`--dataset N` generates N images and pools every well into
`calibration_pairs.csv` (concentration ↔ measured grey, with SNR, saturation
and reliability flags) plus a `dataset_manifest.json`. Three properties make
the pooled data usable rather than merely large:

- **One exposure gain for the whole batch.** Per-image normalization would give
  each image its own vertical axis; the pooled scatter would be several curves
  that never meet.
- **Real blank wells** (concentration exactly 0). Without a blank there is no
  background, and LOD is defined by the blank's mean and spread.
- **Images are gated on localization accuracy against the ground truth.** A
  one-pitch shift silently pairs reading *k* with well *k+1*; the scatter plot
  looks fine and the curve comes out wrong. Rejected images are listed with
  their reason in the manifest.

The requested range is clamped to what the exposure can actually resolve. The
floor is set by **substrate autofluorescence plus filter leakage** (≈0.015
full-well), an order of magnitude above read noise (≈5×10⁻⁴) — wells below it
are not merely noisy, they are invisible, and a third of the array going dark
makes lattice fitting drop rows entirely.

```powershell
python pg_fluoro_sim.py --grid 15 --concentration --dataset 8 --output sim/dataset
```

Measured on that command (8 images, 15×15, requested 1 nM–100 µM):

| | |
|---|---|
| Effective range after clamping | 0.470 – 100 µM |
| Images passing localization | 8/8, 1800 pooled pairs |
| Blank background | −0.03 ± 0.32 grey |
| Detection limit (blank + 3σ) | 3.13 µM |
| Monotonic upper bound | 100 µM — no turnover in range |
| Rank fidelity over the usable range | 0.987 |

Per-preset yield at 15×15, 3 images each: `ideal` 3/3, `typical` 3/3, `hard`
3/3, `extreme` 2/3, with 1.3–3.7 %pitch localization error on the passing
images.

**10×10 used to fail systematically here, and the explanation recorded in this
file was wrong.** The symptom was that sharper scenes did worse than blurred
ones — `ideal` (defocus 0.6) collapsed while `hard` (defocus 2.2) worked — and
that was read as "razor-sharp synthetic squares are harder for the blob
detector." The actual cause was a `max_area` ceiling that scaled with the
canvas side instead of the unit pitch, so a normal 10×10 unit was rejected for
being *too large*; defocus rescued the case by shrinking each unit's
above-threshold core until it slipped back under the ceiling. Both detectors'
gates are now anchored to the 15×15 reference and converted by unit pitch, and
the `--sweep` numbers below are what the pipeline actually does.

`plate_series` assigns one concentration per row (replicates within a row give
the calibration curve its error bars) and **shuffles the row order**. Both
reasons matter: a monotonic concentration gradient across the plate is fully
confounded with the illumination gradient and vignetting, and it also bands
the dimmest wells along one edge, where lattice fitting completes the grid on
the wrong side — measured at ~110 %pitch mean error on 15×15 before the
shuffle.

One caveat is recorded in the manifest rather than left to be discovered:
image rejection is not random. It favors layouts with dim rows near the edge,
and edge wells are more strongly vignetted, so surviving images run slightly
bright at the low end and the LOD is optimistic. Raising N dilutes it; nothing
short of abandoning end-to-end evaluation removes it.

`--intensity-scan` answers the practical question directly: it steps the
overall emission level down and reports where localization stops working. On
the default 15×15 configuration the pipeline is reliable down to a peak of
about 20 grey levels (support 0.71, localization 1.98 %pitch, rank fidelity
0.989) and fails cleanly below roughly 10 grey levels — flagging
`trusted=false` rather than returning a wrong grid. 10×10 now holds the same
envelope: 0.33–0.50 %pitch from 166 down to 41 grey levels, 1.82 %pitch with
rank fidelity 0.987 at 20 grey levels, `trusted=false` from 10 grey levels
down.

The trust flag is fully correct on this scan — true for all four working
levels, false for all five failing ones — which locates the false-alarm
problem described below precisely: it appears only when a single image spans a
wide intensity range. Scaling the whole array together keeps support at
0.74–1.00, whereas a dilution series leaves a fixed fraction of wells under
the floor and caps support near 0.55 regardless of how good the geometry is.

### Localization accuracy baseline

Mean localization error against the simulator's ground truth, from
`python pg_fluoro_sim.py --grid {10,15} --output <dir> --sweep`:

| preset | 10×10 | 15×15 |
|---|---|---|
| `ideal` | 1.46 %pitch | 2.00 %pitch |
| `typical` | 2.71 %pitch | 4.19 %pitch |
| `hard` | 3.96 %pitch | 103.50 %pitch |
| `extreme` | 5.33 %pitch | 9.96 %pitch |

Error is dominated by the wells that are physically invisible rather than by
geometry: broken out by intensity decade, wells above 0.1 full-well localize to
0.43–0.84 %pitch with rank fidelity ≥0.99, while wells in the 0.001–0.01 decade
— at or below the substrate autofluorescence floor — contribute 2–14 %pitch.

Two known defects remain, both visible in that table:

- **15×15 `hard` is a whole-lattice one-pitch row shift.** Row *r* of the fitted
  grid sits on ground-truth row *r+1*: the true top row is never claimed and a
  phantom row is invented one pitch past the bottom edge. Comparing against
  `truth[r+1][c]` collapses the residual from 103.50 to 4.93 %pitch, so the
  geometry is right and only the index assignment is off. All three arbitration
  signals are ratios over the shared rows, so none of them move — coverage and
  support are bit-identical between the wrong grid and the corrected one. The
  run does report `trusted=false`, so the never-fail-silently contract holds,
  but it holds by luck rather than by detection.
- **The trust flag cries wolf.** `candidate_support_ratio` is bounded above by
  the fraction of wells that are physically visible; across the sweep and scan
  runs it lands at a near-constant 0.71–0.75 of that fraction, and a log
  dilution series leaves only ~0.72–0.77 of wells above the floor. Support
  therefore settles around 0.52–0.57, just under the 0.6 gate, for runs that
  localize to 1.5–4 %pitch. Measured on `examples/fluo/`, 8 of the 18 correctly
  localized cases are flagged untrusted while all 6 genuine failures are caught
  — no misses, but a 44% false-alarm rate.

These interact, which fixes the order of work: the support gate cannot simply be
lowered, because at 0.55 it cannot distinguish 10×10 `ideal` (1.46 %pitch) from
15×15 `hard` (103.50 %pitch) — they report the same support. An evidence-based
shift detector has to land first; only then can the gate be recalibrated
without opening a silent-failure hole.

A lattice-anisotropy test (row pitch versus column pitch, which the one-pitch
shift distorts by 9.9%) was evaluated for that role and **rejected by
measurement**: the real 10×10 photographs reach 0.08–0.11 anisotropy while
localizing correctly, because connectors and channel lines pull the region box
out of alignment with the array, and `partial_region_hard_0006` fails at
123 %pitch with 0.0003 anisotropy because a pure translation preserves both
pitches. No threshold separates the two populations.

Self-luminous arrays have no continuous bright panel, so region detection
degrades through two dedicated paths, ordered by how reliable their geometric
reference is:

1. **Faint substrate contour** (preferred): substrate autofluorescence is often
   only 1-2 grey levels above the dark background — invisible per pixel, but it
   covers hundreds of thousands of contiguous pixels. A morphological opening
   with a structuring element larger than one well erases the bright emitters
   and leaves the substrate as a plateau that thresholds cleanly. Crucially this
   reference is *independent of the well intensity distribution*. (Gaussian
   low-pass does **not** work here — it smears emitter energy onto the
   substrate and destroys the very contrast being measured: measured 0/24 hits
   versus 22/24 for opening on the bundled failure cases.)
2. **Emitting dot cloud**: only lit wells contribute, so when whole rows of dim
   wells go undetected the box collapses toward the bright side. Outliers are
   rejected by nearest-neighbour distance — hot pixels are isolated while array
   points sit one pitch apart.

These hypotheses are **not** applied in a fixed priority order. Each one is
carried all the way through rectification and lattice fitting, and the winner is
chosen by evidence — the same "generate several hypotheses, let evidence
arbitrate" pattern this pipeline already uses for the region threshold and for
unit polarity. The scoring combines three complementary signals:

- **candidate support**: what fraction of grid points sit on a real detected
  unit — catches a grid that is displaced or invented;
- **grid coverage**: what fraction of units detected *in a widened field of
  view* are covered by the grid — the reverse direction, and the only one that
  catches a region box that truncated part of the array (truncated units are
  still visible in the widened view but have no grid point on them);
- **observed ratio**: what fraction of points obtained local image evidence.

Coverage is measured on an expanded rectification (same homography, larger
canvas) so units just outside the region box enter the field of view while
their pixel size — and therefore the detector's geometric gates — stay
unchanged. Note its absolute value is layout-dependent (real 10x10 boards with
connectors and channel lines sit at 0.84-0.90 while clean 15x15 boards reach
0.99-1.00), so it drives *relative* comparison between hypotheses and only a
deliberately loose absolute trust gate.

`examples/fluo/` holds a set of real failure inputs with ground truth, kept as a
permanent regression test. Its central contract is that the pipeline must never
fail *silently*: localization may fail on physically hopeless inputs (wells
below the quantization floor, many dead wells), but such runs must report
`trusted=false`. This matters because the candidate-support ratio is blind to
whole-lattice shifts by an integer pitch — a grid displaced by exactly one
pitch still has 14/15 of its points sitting on real wells.

## Colorimetric Simulation with Ground Truth

`pg_colori_sim.py` is the transmission counterpart of the fluorescence
simulator, matching the v4.2 hardware's colorimetric mode: a white EL panel
backlights the chip through a black PMMA mask (15×15 square apertures, 0.5 mm
on a 1.0 mm pitch), and the phone photographs the array from above.

Geometrically this looks like the fluorescence case — bright wells on a dark
field, so the pipeline still detects `polarity=bright`. Physically it is the
inverse: wells get **darker** with concentration, and the measured quantity is
`A = -log10(I_sample / I_blank)` rather than an emission level. Everything
downstream of scene radiance — geometry, defocus, vignetting, sensor noise,
quantization — is shared with `pg_fluoro_sim` through `apply_optics_and_sensor`
so the two modes cannot drift apart in what they claim about noise or SNR.

```powershell
python pg_colori_sim.py --grid 15 --output sim/c15 --evaluate
python pg_colori_sim.py --grid 15 --output sim/csweep --sweep
python pg_colori_sim.py --output . --linearity
```

Absorbance is integrated per wavelength rather than given a scalar `ε` per
channel, because **Beer-Lambert does not commute with spectral integration**:

```
I_ch = ∫ S(λ)·R_ch(λ)·10^(−ε(λ)·c·l) dλ
```

A camera channel is tens of nanometres wide, so at high concentration the
integral is dominated by the wavelengths where `ε` is *smallest*, and apparent
absorbance bends away from linearity. This is the colorimetric analogue of the
fluorescence inner-filter turnover, and it sets the upper end of the working
range. A scalar-`ε` model produces a perfectly straight line and hides it.

Because both a blank and a sample are needed, `write_colorimetric_sample`
always emits a paired blank image rendered from the same seed — the fixed
pattern noise then cancels in the ratio, exactly as it does on real hardware.

### The 0.5 mm liquid layer sets the working range

The chip is 20×20×0.5 mm, so the optical path is 0.5 mm — 20× shorter than a
standard cuvette and 6× shorter than a microplate well. With TMB's acid
endpoint (ε ≈ 5.9×10⁴ M⁻¹cm⁻¹ at 450 nm), measured on the blue channel:

| Target | Concentration needed |
|---|---|
| A = 0.1 (practical floor) | 47 µM |
| A = 1.0 | 471 µM |
| A = 2.0 | 943 µM |

Polychromatic compression is already 5% at A ≈ 0.4 (200 µM) and 22% at
A ≈ 1.2 (700 µM), so the usefully linear range is roughly **25–200 µM**, with
250–700 µM usable only against a non-linear calibration curve. Sub-µM
colorimetric detection is not reachable with this chip geometry; the
fluorescence mode's ~3 µM LOD is the better route for low concentrations.
Path length is the single most effective lever if colorimetric sensitivity
needs to improve.

Measured on `--sweep` at 15×15:

| preset | localization | A_B Spearman | blank A_B | trusted |
|---|---|---|---|---|
| `ideal` | 1.27 %pitch | 0.9995 | 0.0003 ± 0.0012 | true |
| `typical` | 1.34 %pitch | 0.9995 | 0.0000 ± 0.0014 | true |
| `hard` | 215.01 %pitch | −0.12 | — | **true (wrong)** |
| `extreme` | 236.46 %pitch | 0.57 | — | false |

Channel selectivity behaves as it should: for the yellow product the blue
channel reaches A = 1.34 while red only reaches 0.047, so the readout carries
colour information rather than just brightness.

### Colorimetry makes the whole-lattice shift blind spot worse

The `hard` row above is a silent failure — the contract violation the fluo
regression set exists to prevent. It is the same defect documented earlier: an
exhaustive integer-offset search shows that shifting the fitted grid by
(−2 rows, −1 column) collapses the error from 215.01 to **1.82 %pitch**, so the
geometry is right and only the index assignment is wrong.

What is new is that colorimetry removes the accidental protection fluorescence
had. A dilution series leaves a third of its wells under the detection floor,
which drags `candidate_support_ratio` down to ~0.55 and trips the 0.6 gate more
or less by luck. In transmission every well is visible — the span is only about
20:1 — so a displaced grid still lands on real wells and all three signals stay
high: support 0.80, coverage 0.83, observed 0.87. The run is reported as
trusted.

This raises the priority of the evidence-based shift detector: without it, the
colorimetric mode can return a confidently wrong grid, and unlike the
fluorescence case nothing else catches it.

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
