# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-06T14:56:54Z`
- Workflows: `5`
- Warm-start workflows: `3`
- Avg cold mean seconds: `0.615926`
- Avg warm mean seconds: `0.112254`
- Avg warm speedup: `10.006`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.795049` s
- Cold throughput: `0.358` runs/s
- Warm mean: `0.304904` s
- Warm throughput: `3.28` runs/s
- Warm speedup: `9.167`x
- Metadata: `instrument_count`=2, `multi_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'rate_index': 'USD-SOFR-3M'}, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `sabr` single_smile
- Cold mean: `0.041607` s
- Cold throughput: `24.034` runs/s
- Warm mean: `0.003315` s
- Warm throughput: `301.641` runs/s
- Warm speedup: `12.55`x
- Metadata: `point_count`=7, `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit

### `heston` single_smile
- Cold mean: `0.23695` s
- Cold throughput: `4.22` runs/s
- Warm mean: `0.028544` s
- Warm throughput: `35.034` runs/s
- Warm speedup: `8.301`x
- Metadata: `point_count`=5, `warm_start`=True
- Note: least_squares
- Note: fft_pricing

### `local_vol` dupire_surface
- Cold mean: `0.005727` s
- Cold throughput: `174.611` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[15, 30], `warm_start`=False
- Note: dupire
- Note: workflow_surface

### `credit` single_name_curve
- Cold mean: `0.000297` s
- Cold throughput: `3364.485` runs/s
- Warm start: `n/a`
- Metadata: `curve_name`='usd_ig', `model_consistency_contract_version`='v1', `point_count`=4, `quote_family`='spread', `warm_start`=False
- Note: least_squares
- Note: model_consistency_contract_fixture
