# Representative 15x15 Emitting-Grid Localization Failures

This directory contains eight synthetic image cases selected from the local
evaluation set because the current PG-Grid V2.0 pipeline produces a clearly
incorrect 15x15 localization result. The cases are intended only for computer
vision algorithm development: region detection, polarity selection, geometric
grid fitting, local center refinement, confidence calibration, and quality
control.

## Case Summary

| Case | Main failure mode | Trusted | Full coverage | Mean error (% pitch) | P90 error (% pitch) |
| --- | --- | ---: | ---: | ---: | ---: |
| `trusted_wrong_typical_0024` | Wrong dark polarity and incomplete region, but marked trusted | yes | no | 736.66 | 1315.89 |
| `trusted_wrong_gradient_0038` | High candidate support while the fitted grid is globally displaced | yes | yes | 168.18 | 170.62 |
| `trusted_wrong_hard_0019` | Plausible bright grid accepted about one pitch away from truth | yes | yes | 100.38 | 111.20 |
| `trusted_wrong_ideal_0010` | Easy-looking image accepted about one pitch away from truth | yes | yes | 98.39 | 100.23 |
| `region_collapse_extreme_0040` | Chip-region estimate collapses far away from the true panel | no | no | 8392.17 | 12523.26 |
| `partial_region_ideal_0008` | Region detector encloses only a local bright subset | no | no | 695.85 | 1252.21 |
| `checker_failure_0048` | Alternating weak cells create an incorrect lattice phase | no | yes | 251.84 | 441.97 |
| `random_failure_0010` | Sparse random bright cells produce an incorrect lattice hypothesis | no | yes | 223.02 | 354.94 |

Errors are Euclidean center errors normalized by the ground-truth grid pitch.
The metrics were produced by the current `main` implementation at commit
`9bba53e`.

## Directory Layout

Each case contains:

```text
<case_name>/
  input.jpg                   # Original algorithm input
  ground_truth.json           # Ordered 15x15 center coordinates and generator settings
  ground_truth_overlay.jpg    # Ground-truth centers drawn in green on the input
  metrics.json                # Evaluation metrics and failure labels
  current_output/
    overlay_grid.jpg          # Current predicted grid in original-image coordinates
    rectified_chip.jpg        # Region selected and rectified by the current detector
    roi_debug.jpg             # Predicted centers and ROI boxes in rectified coordinates
    result.json               # Localization result and quality-control output
```

`ground_truth.json` stores the 225 points in row-major order. Predicted points
in `current_output/result.json` use the same row-major convention, but their
coordinates are in the rectified image. For original-image inspection, compare
`ground_truth_overlay.jpg` with `current_output/overlay_grid.jpg`.

## Reproduce One Current Output

From the repository root:

```bash
python pg_grid_demo.py \
  --image examples/fluo/trusted_wrong_typical_0024/input.jpg \
  --grid 15 \
  --output outputs/reproduce_typical_0024
```

When optimizing the algorithm, prioritize these invariants:

1. A trusted result must have strong independent geometric evidence, not only a
   high internal fit score.
2. Grid phase must remain correct under gradients, alternating intensity, and
   sparse active cells.
3. Region detection must cover the complete panel before perspective
   rectification.
4. Bright/dark polarity selection must be validated against lattice support and
   image-level contrast evidence.
5. Quality control should reject large geometric errors even when 225 points can
   be generated from a mathematically regular model.
