# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-05T00:19:22Z`
- Workflows: `4`
- Warm-start workflows: `3`
- Avg cold mean seconds: `0.759798`
- Avg warm mean seconds: `0.110528`
- Avg warm speedup: `9.067`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.770749` s
- Cold throughput: `0.361` runs/s
- Warm mean: `0.300092` s
- Warm throughput: `3.332` runs/s
- Warm speedup: `9.233`x
- Metadata: `instrument_count`=2, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `sabr` single_smile
- Cold mean: `0.029773` s
- Cold throughput: `33.587` runs/s
- Warm mean: `0.003046` s
- Warm throughput: `328.261` runs/s
- Warm speedup: `9.773`x
- Metadata: `point_count`=7, `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit

### `heston` single_smile
- Cold mean: `0.233085` s
- Cold throughput: `4.29` runs/s
- Warm mean: `0.028447` s
- Warm throughput: `35.153` runs/s
- Warm speedup: `8.194`x
- Metadata: `point_count`=5, `warm_start`=True
- Note: least_squares
- Note: fft_pricing

### `local_vol` dupire_surface
- Cold mean: `0.005586` s
- Cold throughput: `179.031` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[15, 30], `warm_start`=False
- Note: dupire
- Note: workflow_surface
