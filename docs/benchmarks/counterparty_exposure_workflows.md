# Counterparty Exposure Benchmark: `counterparty_exposure_workflows`
- Created at: `2026-05-07T03:21:30Z`
- Workflows: `2`
- Steady-state workflows: `2`
- Avg cold mean seconds: `0.005885`
- Avg steady mean seconds: `0.002997`
- Avg steady speedup: `11.027`x

## Environment
- Python: `3.10.6`
- Platform: `macOS-26.3.1-arm64-arm-64bit`

## Notes
- Cold runs rebuild the future-value cube and exposure stack; steady-state runs reuse upstream artifacts where the workflow allows it.
- Coverage is limited to supported vanilla IRS future-value cubes, bounded collateral projection, netting-set exposure inputs, and EE/EPE/PFE metrics.

## Workflow Results

### `swap_portfolio_future_value_cube` hull_white_shared_path_irs_book
- Cold mean: `0.005674` s
- Cold throughput: `176.248` runs/s
- Steady mean: `0.005704` s
- Steady throughput: `175.322` runs/s
- Steady speedup: `0.995`x
- Metadata: `n_paths`=96, `n_steps`=36, `position_count`=2, `process_family`='hull_white_1f'
- Note: institutional_exposure
- Note: future_value_cube
- Note: shared_paths

### `counterparty_exposure_metrics` netting_collateral_ee_epe_pfe
- Cold mean: `0.006096` s
- Cold throughput: `164.051` runs/s
- Steady mean: `0.000289` s
- Steady throughput: `3454.723` runs/s
- Steady speedup: `21.059`x
- Metadata: `netting_set_count`=1, `pfe_levels`=[0.95, 0.99], `position_count`=2, `warm_start`='reuse_future_value_cube'
- Note: institutional_exposure
- Note: netting
- Note: collateral
- Note: ee_epe_pfe
