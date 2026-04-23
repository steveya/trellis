# Pod Risk Benchmark: `pod_risk_workflows`
- Created at: `2026-04-23T12:52:58Z`
- Workflows: `6`
- Steady-state workflows: `6`
- Avg cold mean seconds: `0.443488`
- Avg steady mean seconds: `0.375485`
- Avg steady speedup: `3.3`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold runs rebuild sessions or pipelines for each sample, while steady-state runs reuse prebuilt runtime objects when the workflow allows it.
- Coverage includes shared scenario-result cube execution plus supported rates, volatility, and spot-risk analytics.

## Workflow Results

### `pipeline_scenarios` twist_cube
- Cold mean: `0.044221` s
- Cold throughput: `22.614` runs/s
- Steady mean: `0.042102` s
- Steady throughput: `23.752` runs/s
- Steady speedup: `1.05`x
- Metadata: `position_count`=2, `scenario_count`=2
- Note: scenario_cube
- Note: twist_pack

### `key_rate_durations` curve_rebuild_buckets
- Cold mean: `1.62714` s
- Cold throughput: `0.615` runs/s
- Steady mean: `1.434231` s
- Steady throughput: `0.697` runs/s
- Steady speedup: `1.135`x
- Metadata: `bucket_count`=4, `methodology`='curve_rebuild'
- Note: curve_rebuild
- Note: rates_risk

### `portfolio_aad` bond_book_reverse_mode
- Cold mean: `0.035875` s
- Cold throughput: `27.875` runs/s
- Steady mean: `0.002494` s
- Steady throughput: `400.946` runs/s
- Steady speedup: `14.384`x
- Metadata: `curve_nodes`=5, `position_count`=2, `supported_route`='bond_only'
- Note: bond_book
- Note: reverse_mode
- Note: supported_curve

### `scenario_pnl` rebuild_pack
- Cold mean: `0.912669` s
- Cold throughput: `1.096` runs/s
- Steady mean: `0.732731` s
- Steady throughput: `1.365` runs/s
- Steady speedup: `1.246`x
- Metadata: `methodology`='curve_rebuild', `scenario_count`=4
- Note: curve_rebuild
- Note: named_scenarios

### `vega` bucketed_surface
- Cold mean: `0.022658` s
- Cold throughput: `44.135` runs/s
- Steady mean: `0.023035` s
- Steady throughput: `43.412` runs/s
- Steady speedup: `0.984`x
- Metadata: `grid_shape`=[3, 3]
- Note: vol_surface
- Note: bucketed_vega

### `spot_greeks` delta_gamma_theta_bundle
- Cold mean: `0.018363` s
- Cold throughput: `54.456` runs/s
- Steady mean: `0.018316` s
- Steady throughput: `54.597` runs/s
- Steady speedup: `1.003`x
- Metadata: `measure_count`=3
- Note: spot_risk
- Note: bundle
