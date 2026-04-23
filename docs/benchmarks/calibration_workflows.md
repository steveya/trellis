# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-22T23:58:59Z`
- Workflows: `7`
- Warm-start workflows: `4`
- Avg cold mean seconds: `1.26601`
- Avg warm mean seconds: `0.146518`
- Avg warm speedup: `13.383`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.733829` s
- Cold throughput: `0.366` runs/s
- Warm mean: `0.296563` s
- Warm throughput: `3.372` runs/s
- Warm speedup: `9.218`x
- Metadata: `instrument_count`=2, `multi_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'rate_index': 'USD-SOFR-3M'}, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `sabr` single_smile
- Cold mean: `0.076699` s
- Cold throughput: `13.038` runs/s
- Warm mean: `0.003364` s
- Warm throughput: `297.231` runs/s
- Warm speedup: `22.797`x
- Metadata: `point_count`=7, `surface_name`='usd_rates_smile', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit
- Note: synthetic_generation_contract_fixture

### `equity_vol_surface` repaired_surface_authority
- Cold mean: `3.813041` s
- Cold throughput: `0.262` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: svi_surface
- Note: quote_governance
- Note: synthetic_generation_contract_fixture

### `heston` single_smile
- Cold mean: `0.85612` s
- Cold throughput: `1.168` runs/s
- Warm mean: `0.04616` s
- Warm throughput: `21.664` runs/s
- Warm speedup: `18.547`x
- Metadata: `point_count`=5, `surface_name`='spx_heston_implied_vol', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: synthetic_generation_contract_fixture

### `heston_surface` surface_compression
- Cold mean: `0.713178` s
- Cold throughput: `1.402` runs/s
- Warm mean: `0.239985` s
- Warm throughput: `4.167` runs/s
- Warm speedup: `2.972`x
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: surface_compression
- Note: synthetic_generation_contract_fixture

### `local_vol` dupire_surface
- Cold mean: `0.000352` s
- Cold throughput: `2841.134` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `source_surface_name`='spx_heston_implied_vol', `surface_name`='spx_local_vol', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: dupire
- Note: workflow_surface
- Note: synthetic_generation_contract_fixture

### `credit` single_name_curve
- Cold mean: `0.668853` s
- Cold throughput: `1.495` runs/s
- Warm start: `n/a`
- Metadata: `curve_name`='usd_ig', `model_consistency_contract_version`='v1', `point_count`=4, `quote_family`='spread', `warm_start`=False
- Note: least_squares
- Note: model_consistency_contract_fixture
