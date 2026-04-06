# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-06T18:34:55Z`
- Workflows: `5`
- Warm-start workflows: `3`
- Avg cold mean seconds: `0.681113`
- Avg warm mean seconds: `0.111876`
- Avg warm speedup: `17.699`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.644463` s
- Cold throughput: `0.378` runs/s
- Warm mean: `0.290246` s
- Warm throughput: `3.445` runs/s
- Warm speedup: `9.111`x
- Metadata: `instrument_count`=2, `multi_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'rate_index': 'USD-SOFR-3M'}, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `sabr` single_smile
- Cold mean: `0.091279` s
- Cold throughput: `10.955` runs/s
- Warm mean: `0.003248` s
- Warm throughput: `307.905` runs/s
- Warm speedup: `28.105`x
- Metadata: `point_count`=7, `surface_name`='usd_rates_smile', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit
- Note: synthetic_generation_contract_fixture

### `heston` single_smile
- Cold mean: `0.669084` s
- Cold throughput: `1.495` runs/s
- Warm mean: `0.042135` s
- Warm throughput: `23.733` runs/s
- Warm speedup: `15.88`x
- Metadata: `point_count`=5, `surface_name`='spx_heston_implied_vol', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: synthetic_generation_contract_fixture

### `local_vol` dupire_surface
- Cold mean: `0.000378` s
- Cold throughput: `2644.334` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `source_surface_name`='spx_heston_implied_vol', `surface_name`='spx_local_vol', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: dupire
- Note: workflow_surface
- Note: synthetic_generation_contract_fixture

### `credit` single_name_curve
- Cold mean: `0.000362` s
- Cold throughput: `2764.019` runs/s
- Warm start: `n/a`
- Metadata: `curve_name`='usd_ig', `model_consistency_contract_version`='v1', `point_count`=4, `quote_family`='spread', `warm_start`=False
- Note: least_squares
- Note: model_consistency_contract_fixture
