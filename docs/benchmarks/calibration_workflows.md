# Calibration Benchmark: `supported_calibration_workflows`
- Created at: `2026-04-23T23:32:54Z`
- Workflows: `11`
- Warm-start workflows: `5`
- Desk-like workflows: `2`
- Perturbation diagnostics: `2`
- Latency envelopes: `2`
- Avg cold mean seconds: `1.120803`
- Avg warm mean seconds: `0.146099`
- Avg warm speedup: `11.086`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold-start and warm-start runs share the same synthetic benchmark fixtures.
- Warm-start baselines use workflow-native seed hooks where the workflow supports them.

## Workflow Results

### `hull_white` swaption_strip
- Cold mean: `2.718538` s
- Cold throughput: `0.368` runs/s
- Warm mean: `0.292813` s
- Warm throughput: `3.415` runs/s
- Warm speedup: `9.284`x
- Metadata: `instrument_count`=2, `multi_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'rate_index': 'USD-SOFR-3M'}, `warm_start`=True
- Note: least_squares
- Note: tree_pricing

### `caplet_strip` price_bootstrap_surface
- Cold mean: `0.017735` s
- Cold throughput: `56.386` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[4, 2], `quote_count`=8, `surface_name`='usd_caplet_strip', `warm_start`=False
- Note: bootstrap
- Note: caplet_surface
- Note: price_quotes

### `sabr` single_smile
- Cold mean: `0.092521` s
- Cold throughput: `10.808` runs/s
- Warm mean: `0.003399` s
- Warm throughput: `294.193` runs/s
- Warm speedup: `27.219`x
- Metadata: `point_count`=7, `surface_name`='usd_rates_smile', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: implied_vol_fit
- Note: synthetic_generation_contract_fixture

### `swaption_cube` price_normalized_cube
- Cold mean: `0.052409` s
- Cold throughput: `19.081` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[2, 2, 3], `quote_count`=12, `surface_name`='usd_swaption_cube', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: cube_assembly
- Note: swaption_surface
- Note: price_quotes
- Note: synthetic_generation_contract_fixture

### `equity_vol_surface` repaired_surface_authority
- Cold mean: `3.804943` s
- Cold throughput: `0.263` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: svi_surface
- Note: quote_governance
- Note: synthetic_generation_contract_fixture

### `heston` single_smile
- Cold mean: `0.861616` s
- Cold throughput: `1.161` runs/s
- Warm mean: `0.068261` s
- Warm throughput: `14.65` runs/s
- Warm speedup: `12.622`x
- Metadata: `point_count`=5, `surface_name`='spx_heston_implied_vol', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: synthetic_generation_contract_fixture

### `heston_surface` surface_compression
- Cold mean: `1.016551` s
- Cold throughput: `0.984` runs/s
- Warm mean: `0.363279` s
- Warm throughput: `2.753` runs/s
- Warm speedup: `2.798`x
- Metadata: `grid_shape`=[5, 5], `surface_name`='spx_surface_authority', `synthetic_generation_contract_version`='v2', `warm_start`=True
- Note: least_squares
- Note: fft_pricing
- Note: surface_compression
- Note: synthetic_generation_contract_fixture

### `local_vol` dupire_surface
- Cold mean: `0.000395` s
- Cold throughput: `2531.107` runs/s
- Warm start: `n/a`
- Metadata: `grid_shape`=[5, 5], `source_surface_name`='spx_heston_implied_vol', `surface_name`='spx_local_vol', `synthetic_generation_contract_version`='v2', `warm_start`=False
- Note: dupire
- Note: workflow_surface
- Note: synthetic_generation_contract_fixture

### `credit` single_name_curve
- Cold mean: `0.710894` s
- Cold throughput: `1.407` runs/s
- Warm start: `n/a`
- Metadata: `curve_name`='usd_ig', `model_consistency_contract_version`='v1', `point_count`=4, `quote_family`='spread', `warm_start`=False
- Note: least_squares
- Note: model_consistency_contract_fixture

### `basket_credit` desk_tranche_surface
- Cold mean: `3.043611` s
- Cold throughput: `0.329` runs/s
- Warm start: `n/a`
- Latency envelope: `pass` (cold mean `3.043611` s <= `6.0` s)
- Perturbation diagnostic: `pass` (max abs change `0.004268596412018488`)
- Metadata: `fixture_style`='desk_like', `latency_envelope`={'workflow': 'basket_credit', 'label': 'desk_tranche_surface', 'fixture_style': 'desk_like', 'instrument_count': None, 'quote_count': 6, 'cold_mean_limit_seconds': 6.0, 'cold_max_limit_seconds': 8.0, 'warm_mean_limit_seconds': None}, `linked_credit_curve`='benchmark_single_name_credit', `maturity_count`=2, `perturbation_diagnostic`={'label': 'basket_credit_parallel_quote_up', 'perturbation_size': 0.0025, 'baseline_metrics': {'5y_0.00_0.03': 0.18000000000001112, '5y_0.03_0.07': 0.23999999999990046, '5y_0.07_0.10': 0.33999999999225816, '7y_0.00_0.03': 0.39, '7y_0.03_0.07': 0.48000000000000054, '7y_0.07_0.10': 0.5399999999999684}, 'perturbed_metrics': {'5y_0.00_0.03': 0.1781832312816442, '5y_0.03_0.07': 0.23573140358788197, '5y_0.07_0.10': 0.34400870869280414, '7y_0.00_0.03': 0.3880026451908997, '7y_0.03_0.07': 0.4774988971374956, '7y_0.07_0.10': 0.5361447035161097}, 'absolute_changes': {'5y_0.00_0.03': -0.0018167687183669179, '5y_0.03_0.07': -0.004268596412018488, '5y_0.07_0.10': 0.004008708700545982, '7y_0.00_0.03': -0.0019973548091002935, '7y_0.03_0.07': -0.0025011028625049336, '7y_0.07_0.10': -0.003855296483858739}, 'relative_changes': {'5y_0.00_0.03': -0.010093159546482253, '5y_0.03_0.07': -0.017785818383417744, '5y_0.07_0.10': 0.011790319707756649, '7y_0.00_0.03': -0.00512142258743665, '7y_0.03_0.07': -0.00521063096355194, '7y_0.07_0.10': -0.007139437933072156}, 'max_abs_change': 0.004268596412018488, 'max_relative_change': 0.017785818383417744, 'threshold_breaches': {}, 'status': 'pass'}, `quote_count`=6, `support_boundary`='homogeneous_representative_curve', `surface_name`='benchmark_tranche_correlation', `tranche_count`=3, `warm_start`=False
- Note: brentq_root_scan
- Note: homogeneous_basket_credit
- Note: desk_like_fixture
- Note: linked_single_name_credit_curve

### `quanto_correlation` desk_quanto_correlation
- Cold mean: `0.00962` s
- Cold throughput: `103.952` runs/s
- Warm mean: `0.002744` s
- Warm throughput: `364.42` runs/s
- Warm speedup: `3.506`x
- Latency envelope: `pass` (cold mean `0.00962` s <= `1.5` s)
- Perturbation diagnostic: `pass` (max abs change `0.00741841223079881`)
- Metadata: `correlation_keys`=['EURUSD_corr'], `fixture_style`='desk_like', `fx_pair`='EURUSD', `latency_envelope`={'workflow': 'quanto_correlation', 'label': 'desk_quanto_correlation', 'fixture_style': 'desk_like', 'instrument_count': None, 'quote_count': 3, 'cold_mean_limit_seconds': 1.5, 'cold_max_limit_seconds': 2.0, 'warm_mean_limit_seconds': 0.5}, `linked_curve_roles`={'discount_curve': 'usd_ois', 'forecast_curve': 'EUR-DISC'}, `linked_vol_surface`='quanto_flat_vol', `parameter_set_name`='benchmark_quanto_rho', `perturbation_diagnostic`={'label': 'quanto_correlation_parallel_quote_up', 'perturbation_size': 0.0025, 'baseline_metrics': {'quanto_correlation': 0.3499999999999991}, 'perturbed_metrics': {'quanto_correlation': 0.3425815877692003}, 'absolute_changes': {'quanto_correlation': -0.00741841223079881}, 'relative_changes': {'quanto_correlation': -0.021195463516568085}, 'max_abs_change': 0.00741841223079881, 'max_relative_change': 0.021195463516568085, 'threshold_breaches': {}, 'status': 'pass'}, `quote_count`=3, `support_boundary`='bounded_quanto_correlation', `warm_start`=True
- Note: least_squares
- Note: desk_like_fixture
- Note: bounded_quanto_correlation
- Note: linked_market_state_materialization
