# Pod Risk Benchmark: `pod_risk_workflows`
- Created at: `2026-04-05T19:51:03Z`
- Workflows: `5`
- Steady-state workflows: `5`
- Avg cold mean seconds: `0.49353`
- Avg steady mean seconds: `0.418257`
- Avg steady speedup: `1.093`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold runs rebuild sessions or pipelines for each sample, while steady-state runs reuse prebuilt runtime objects when the workflow allows it.
- Coverage includes shared scenario-result cube execution plus supported rates, volatility, and spot-risk analytics.

## Workflow Results

### `pipeline_scenarios` twist_cube
- Cold mean: `0.036345` s
- Cold throughput: `27.514` runs/s
- Steady mean: `0.033873` s
- Steady throughput: `29.522` runs/s
- Steady speedup: `1.073`x
- Metadata: `position_count`=2, `scenario_count`=2
- Note: scenario_cube
- Note: twist_pack

### `key_rate_durations` curve_rebuild_buckets
- Cold mean: `1.553535` s
- Cold throughput: `0.644` runs/s
- Steady mean: `1.35` s
- Steady throughput: `0.741` runs/s
- Steady speedup: `1.151`x
- Metadata: `bucket_count`=4, `methodology`='curve_rebuild'
- Note: curve_rebuild
- Note: rates_risk

### `scenario_pnl` rebuild_pack
- Cold mean: `0.846134` s
- Cold throughput: `1.182` runs/s
- Steady mean: `0.675639` s
- Steady throughput: `1.48` runs/s
- Steady speedup: `1.252`x
- Metadata: `methodology`='curve_rebuild', `scenario_count`=4
- Note: curve_rebuild
- Note: named_scenarios

### `vega` bucketed_surface
- Cold mean: `0.018114` s
- Cold throughput: `55.205` runs/s
- Steady mean: `0.017966` s
- Steady throughput: `55.662` runs/s
- Steady speedup: `1.008`x
- Metadata: `grid_shape`=[3, 3]
- Note: vol_surface
- Note: bucketed_vega

### `spot_greeks` delta_gamma_theta_bundle
- Cold mean: `0.013524` s
- Cold throughput: `73.942` runs/s
- Steady mean: `0.013805` s
- Steady throughput: `72.44` runs/s
- Steady speedup: `0.98`x
- Metadata: `measure_count`=3
- Note: spot_risk
- Note: bundle
