# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-23T04:31:29Z`
- Workflows: `10`
- Warm-start workflows: `4`
- Desk-like workflows: `1`
- Perturbation diagnostics: `1`
- Latency envelopes: `1`
- Avg cold mean seconds: `1.219831`
- Avg warm mean seconds: `0.179496`
- Avg warm speedup: `12.025`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.676626` s
- Cold throughput: `0.374` runs/s
- Warm mean: `0.291412` s
- Warm throughput: `3.432` runs/s
- Warm speedup: `9.185`x
- Metadata: `instrument_count`=2, `multi_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'rate_index': 'USD-SOFR-3M'}, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `caplet_strip` price_bootstrap_surface
- Cold mean: `0.016651` s
- Cold throughput: `60.055` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[4, 2], `quote_count`=8, `surface_name`='usd_caplet_strip', `warm_start`=False
- Note: bootstrap
- Note: caplet_surface
- Note: price_quotes

### `sabr` single_smile
- Cold mean: `0.072777` s
- Cold throughput: `13.741` runs/s
- Warm mean: `0.003105` s
- Warm throughput: `322.112` runs/s
- Warm speedup: `23.442`x
- Metadata: `point_count`=7, `surface_name`='usd_rates_smile', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit
- Note: synthetic_generation_contract_fixture

### `swaption_cube` price_normalized_cube
- Cold mean: `0.04977` s
- Cold throughput: `20.092` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[2, 2, 3], `quote_count`=12, `surface_name`='usd_swaption_cube', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: cube_assembly
- Note: swaption_surface
- Note: price_quotes
- Note: synthetic_generation_contract_fixture

### `equity_vol_surface` repaired_surface_authority
- Cold mean: `3.723247` s
- Cold throughput: `0.269` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: svi_surface
- Note: quote_governance
- Note: synthetic_generation_contract_fixture

### `heston` single_smile
- Cold mean: `0.862357` s
- Cold throughput: `1.16` runs/s
- Warm mean: `0.068199` s
- Warm throughput: `14.663` runs/s
- Warm speedup: `12.645`x
- Metadata: `point_count`=5, `surface_name`='spx_heston_implied_vol', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: synthetic_generation_contract_fixture

### `heston_surface` surface_compression
- Cold mean: `1.004092` s
- Cold throughput: `0.996` runs/s
- Warm mean: `0.355267` s
- Warm throughput: `2.815` runs/s
- Warm speedup: `2.826`x
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: surface_compression
- Note: synthetic_generation_contract_fixture

### `local_vol` dupire_surface
- Cold mean: `0.000341` s
- Cold throughput: `2933.747` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `source_surface_name`='spx_heston_implied_vol', `surface_name`='spx_local_vol', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: dupire
- Note: workflow_surface
- Note: synthetic_generation_contract_fixture

### `credit` single_name_curve
- Cold mean: `0.695911` s
- Cold throughput: `1.437` runs/s
- Warm start: `n/a`
- Metadata: `curve_name`='usd_ig', `model_consistency_contract_version`='v1', `point_count`=4, `quote_family`='spread', `warm_start`=False
- Note: least_squares
- Note: model_consistency_contract_fixture

### `basket_credit` desk_tranche_surface
- Cold mean: `3.096537` s
- Cold throughput: `0.323` runs/s
- Warm start: `n/a`
- Latency envelope: `pass` (cold mean `3.096537` s <= `6.0` s)
- Perturbation diagnostic: `pass` (max abs change `0.004268596412018488`)
- Metadata: `fixture_style`='desk_like', `latency_envelope`={'workflow': 'basket_credit', 'label': 'desk_tranche_surface', 'fixture_style': 'desk_like', 'instrument_count': None, 'quote_count': 6, 'cold_mean_limit_seconds': 6.0, 'cold_max_limit_seconds': 8.0, 'warm_mean_limit_seconds': None}, `linked_credit_curve`='benchmark_single_name_credit', `maturity_count`=2, `perturbation_diagnostic`={'label': 'basket_credit_parallel_quote_up', 'perturbation_size': 0.0025, 'baseline_metrics': {'5y_0.00_0.03': 0.18000000000001112, '5y_0.03_0.07': 0.23999999999990046, '5y_0.07_0.10': 0.33999999999225816, '7y_0.00_0.03': 0.39, '7y_0.03_0.07': 0.48000000000000054, '7y_0.07_0.10': 0.5399999999999684}, 'perturbed_metrics': {'5y_0.00_0.03': 0.1781832312816442, '5y_0.03_0.07': 0.23573140358788197, '5y_0.07_0.10': 0.34400870869280414, '7y_0.00_0.03': 0.3880026451908997, '7y_0.03_0.07': 0.4774988971374956, '7y_0.07_0.10': 0.5361447035161097}, 'absolute_changes': {'5y_0.00_0.03': -0.0018167687183669179, '5y_0.03_0.07': -0.004268596412018488, '5y_0.07_0.10': 0.004008708700545982, '7y_0.00_0.03': -0.0019973548091002935, '7y_0.03_0.07': -0.0025011028625049336, '7y_0.07_0.10': -0.003855296483858739}, 'relative_changes': {'5y_0.00_0.03': -0.010093159546482253, '5y_0.03_0.07': -0.017785818383417744, '5y_0.07_0.10': 0.011790319707756649, '7y_0.00_0.03': -0.00512142258743665, '7y_0.03_0.07': -0.00521063096355194, '7y_0.07_0.10': -0.007139437933072156}, 'max_abs_change': 0.004268596412018488, 'max_relative_change': 0.017785818383417744, 'threshold_breaches': {}, 'status': 'pass'}, `quote_count`=6, `support_boundary`='homogeneous_representative_curve', `surface_name`='benchmark_tranche_correlation', `tranche_count`=3, `warm_start`=False
- Note: brentq_root_scan
- Note: homogeneous_basket_credit
- Note: desk_like_fixture
- Note: linked_single_name_credit_curve
