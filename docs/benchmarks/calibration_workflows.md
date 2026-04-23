# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-23T00:30:26Z`
- Workflows: `9`
- Warm-start workflows: `4`
- Avg cold mean seconds: `0.994036`
- Avg warm mean seconds: `0.148439`
- Avg warm speedup: `13.471`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.726097` s
- Cold throughput: `0.367` runs/s
- Warm mean: `0.298242` s
- Warm throughput: `3.353` runs/s
- Warm speedup: `9.141`x
- Metadata: `instrument_count`=2, `multi_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'rate_index': 'USD-SOFR-3M'}, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `caplet_strip` price_bootstrap_surface
- Cold mean: `0.017628` s
- Cold throughput: `56.729` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[4, 2], `quote_count`=8, `surface_name`='usd_caplet_strip', `warm_start`=False
- Note: bootstrap
- Note: caplet_surface
- Note: price_quotes

### `sabr` single_smile
- Cold mean: `0.07621` s
- Cold throughput: `13.122` runs/s
- Warm mean: `0.00318` s
- Warm throughput: `314.419` runs/s
- Warm speedup: `23.962`x
- Metadata: `point_count`=7, `surface_name`='usd_rates_smile', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit
- Note: synthetic_generation_contract_fixture

### `swaption_cube` price_normalized_cube
- Cold mean: `0.052007` s
- Cold throughput: `19.228` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[2, 2, 3], `quote_count`=12, `surface_name`='usd_swaption_cube', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: cube_assembly
- Note: swaption_surface
- Note: price_quotes
- Note: synthetic_generation_contract_fixture

### `equity_vol_surface` repaired_surface_authority
- Cold mean: `3.84823` s
- Cold throughput: `0.26` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: svi_surface
- Note: quote_governance
- Note: synthetic_generation_contract_fixture

### `heston` single_smile
- Cold mean: `0.847418` s
- Cold throughput: `1.18` runs/s
- Warm mean: `0.047441` s
- Warm throughput: `21.079` runs/s
- Warm speedup: `17.863`x
- Metadata: `point_count`=5, `surface_name`='spx_heston_implied_vol', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: synthetic_generation_contract_fixture

### `heston_surface` surface_compression
- Cold mean: `0.714674` s
- Cold throughput: `1.399` runs/s
- Warm mean: `0.244894` s
- Warm throughput: `4.083` runs/s
- Warm speedup: `2.918`x
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: surface_compression
- Note: synthetic_generation_contract_fixture

### `local_vol` dupire_surface
- Cold mean: `0.000346` s
- Cold throughput: `2886.469` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `source_surface_name`='spx_heston_implied_vol', `surface_name`='spx_local_vol', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: dupire
- Note: workflow_surface
- Note: synthetic_generation_contract_fixture

### `credit` single_name_curve
- Cold mean: `0.663713` s
- Cold throughput: `1.507` runs/s
- Warm start: `n/a`
- Metadata: `curve_name`='usd_ig', `model_consistency_contract_version`='v1', `point_count`=4, `quote_family`='spread', `warm_start`=False
- Note: least_squares
- Note: model_consistency_contract_fixture
