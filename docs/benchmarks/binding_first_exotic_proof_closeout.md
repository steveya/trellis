# Binding-First Exotic Program Closeout

- Tasks: `11`
- Passed gate: `1`
- Failed gate: `10`
- Proved expectations: `10`
- Honest-block expectations: `1`
- Certified honest blocks: `0`
- First-pass success rate: `0.09090909090909091`
- Average attempts to success (successful tasks): `1.0`
- Total elapsed seconds: `1143.1`
- Total tokens: `883606`
- Unknown route-id tasks: `E26, E27, T17, T50`

## Cohorts
### basket_credit_loss
- Status: `failed_gate`
- Passed gate: `0`
- Failed gate: `6`
- Tasks: `T49, T50, T53, E26, T102, T126`
- Report JSON: `qua809_report_live.json`
- Report Markdown: `qua809_report_live.md`

### event_control_schedule
- Status: `failed_gate`
- Passed gate: `1`
- Failed gate: `4`
- Tasks: `T17, T73, T105, E22, E27`
- Report JSON: `qua808_report_live.json`
- Report Markdown: `qua808_report_live.md`

## Task View
### E22 - Cap/floor: Black caplet stack vs MC rate simulation
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo, trellis.models.black.black76_call`
- Route ids: `monte_carlo_paths, analytical_black76`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `code_generation, validation`
- Latest diagnosis dossier: `E22.md`

### E26 - Nth-to-default basket: Gaussian copula vs default-time MC
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.instruments.nth_to_default.price_nth_to_default_basket`
- Route ids: `nth_to_default_monte_carlo, unknown`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `actual_market_smoke`
- Latest diagnosis dossier: `E26.md`

### E27 - American Asian barrier under Heston: PDE vs MC vs FFT should block honestly
- Cohort: `event_control_schedule`
- Expected outcome: `honest_block`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `exercise:exercise:fallback`
- Route ids: `exercise_monte_carlo, unknown`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `E27.md`

### T102 - Rainbow option (best-of-two): Stulz formula vs MC
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.models.black.black76_call, trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo`
- Route ids: `analytical_black76, monte_carlo_paths`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `lite_review, semantic_validation, actual_market_smoke, code_generation`
- Latest diagnosis dossier: `T102.md`

### T105 - Quanto option: quanto-adjusted BS vs MC cross-currency
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.quanto_option.price_quanto_option_analytical_from_market_state, trellis.models.quanto_option.price_quanto_option_monte_carlo_from_market_state`
- Route ids: `quanto_adjustment_analytical, correlated_gbm_monte_carlo`
- First pass: `True`
- Attempts to success: `1`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T105.md`

### T126 - Spread option (Kirk approximation) vs 2D MC vs 2D FFT
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.models.black.black76_call, trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo, fft_pricing:fft_pricing:fallback`
- Route ids: `analytical_black76, monte_carlo_paths, transform_fft`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `semantic_validation, actual_market_smoke`
- Latest diagnosis dossier: `T126.md`

### T17 - Callable bond: HW rate PDE (PSOR) vs HW tree
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.models.callable_bond_tree.price_callable_bond_tree`
- Route ids: `exercise_lattice, unknown`
- First pass: `True`
- Attempts to success: `1`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T17.md`

### T49 - CDO tranche: Gaussian vs Student-t copula
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.models.credit_basket_copula.price_credit_basket_tranche`
- Route ids: `copula_loss_distribution`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `actual_market_smoke`
- Latest diagnosis dossier: `T49.md`

### T50 - Nth-to-default: MC correlated defaults vs semi-analytical
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.instruments.nth_to_default.price_nth_to_default_basket`
- Route ids: `nth_to_default_monte_carlo, unknown`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `actual_market_smoke, semantic_validation`
- Latest diagnosis dossier: `T50.md`

### T53 - Multi-name portfolio loss distribution: recursive vs FFT vs MC
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_insufficient_results`
- Comparison status: `insufficient_results`
- Binding ids: `trellis.models.black.black76_call, fft_pricing:fft_pricing:fallback, trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo`
- Route ids: `analytical_black76, transform_fft, monte_carlo_paths`
- First pass: `False`
- Attempts to success: `3`
- Retry taxonomy: `actual_market_smoke, semantic_validation, code_generation`
- Latest diagnosis dossier: `T53.md`

### T73 - European swaption: Black76 vs HW tree vs HW MC
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `False`
- Failure bucket: `comparison_failed`
- Comparison status: `failed`
- Binding ids: `trellis.models.rate_style_swaption.price_swaption_black76, trellis.models.rate_style_swaption_tree.price_swaption_tree, trellis.models.rate_style_swaption.price_swaption_monte_carlo`
- Route ids: `analytical_black76, rate_tree_backward_induction, monte_carlo_paths`
- First pass: `True`
- Attempts to success: `1`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T73.md`
