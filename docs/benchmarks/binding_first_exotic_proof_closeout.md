# Binding-First Exotic Program Closeout

- Tasks: `11`
- Passed gate: `11`
- Failed gate: `0`
- Proved expectations: `10`
- Honest-block expectations: `1`
- Certified honest blocks: `1`
- First-pass success rate: `0.9090909090909091`
- Average attempts to success (successful tasks): `0.4`
- Total elapsed seconds: `304.6`
- Total tokens: `142551`
- Unknown route-id tasks: `none`

## Failure Buckets
- `blocked`: `1`
- `success`: `10`

## Cohorts
### basket_credit_loss
- Status: `completed`
- Passed gate: `6`
- Failed gate: `0`
- Tasks: `T49, T50, T53, E26, T102, T126`
- Report JSON: `qua825_basket_credit_loss_final_v2_report.json`
- Report Markdown: `qua825_basket_credit_loss_final_v2_report.md`

### event_control_schedule
- Status: `completed`
- Passed gate: `5`
- Failed gate: `0`
- Tasks: `T17, T73, T105, E22, E27`
- Report JSON: `qua819_event_control_schedule_final_report.json`
- Report Markdown: `qua819_event_control_schedule_final_report.md`

## Task View
### E22 - Cap/floor: Black caplet stack vs MC rate simulation
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo, trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical`
- Route ids: `monte_carlo_paths, analytical_black76`
- First pass: `False`
- Attempts to success: `2`
- Retry taxonomy: `code_generation`
- Latest diagnosis dossier: `E22.md`

### E26 - Nth-to-default basket: Gaussian copula vs default-time MC
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.instruments.nth_to_default.price_nth_to_default_basket`
- Route ids: `nth_to_default_analytical, nth_to_default_monte_carlo`
- First pass: `True`
- Attempts to success: `1`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `E26.md`

### E27 - American Asian barrier under Heston: PDE vs MC vs FFT should block honestly
- Cohort: `event_control_schedule`
- Expected outcome: `honest_block`
- Gate passed: `True`
- Failure bucket: `blocked`
- Comparison status: `insufficient_results`
- Binding ids: `exercise:exercise:fallback`
- Route ids: `exercise_monte_carlo`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `E27.md`

### T102 - Rainbow option (best-of-two): Stulz formula vs MC
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.basket_option.price_basket_option_analytical, trellis.models.basket_option.price_basket_option_monte_carlo`
- Route ids: `monte_carlo_paths, analytical_black76`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T102.md`

### T105 - Quanto option: quanto-adjusted BS vs MC cross-currency
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.quanto_option.price_quanto_option_analytical_from_market_state, trellis.models.quanto_option.price_quanto_option_monte_carlo_from_market_state`
- Route ids: `equity_quanto, equity_quanto`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T105.md`

### T126 - Spread option (Kirk approximation) vs 2D MC vs 2D FFT
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.basket_option.price_basket_option_analytical, trellis.models.basket_option.price_basket_option_monte_carlo, trellis.models.basket_option.price_basket_option_transform_proxy`
- Route ids: `monte_carlo_paths, transform_fft, analytical_black76`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T126.md`

### T17 - Callable bond: HW rate PDE (PSOR) vs HW tree
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.callable_bond_pde.price_callable_bond_pde, trellis.models.callable_bond_tree.price_callable_bond_tree`
- Route ids: `pde_theta_1d, exercise_lattice`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T17.md`

### T49 - CDO tranche: Gaussian vs Student-t copula
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.credit_basket_copula.price_credit_basket_tranche`
- Route ids: `copula_loss_distribution`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T49.md`

### T50 - Nth-to-default: MC correlated defaults vs semi-analytical
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.instruments.nth_to_default.price_nth_to_default_basket`
- Route ids: `nth_to_default_monte_carlo, nth_to_default_analytical`
- First pass: `True`
- Attempts to success: `1`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T50.md`

### T53 - Multi-name portfolio loss distribution: recursive vs FFT vs MC
- Cohort: `basket_credit_loss`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_recursive, trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_transform_proxy, trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_monte_carlo`
- Route ids: `copula_loss_distribution, transform_fft, monte_carlo_paths`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T53.md`

### T73 - European swaption: Black76 vs HW tree vs HW MC
- Cohort: `event_control_schedule`
- Expected outcome: `proved`
- Gate passed: `True`
- Failure bucket: `success`
- Comparison status: `passed`
- Binding ids: `trellis.models.rate_style_swaption.price_swaption_black76, trellis.models.rate_style_swaption_tree.price_swaption_tree, trellis.models.rate_style_swaption.price_swaption_monte_carlo`
- Route ids: `analytical_black76, rate_tree_backward_induction, monte_carlo_paths`
- First pass: `True`
- Attempts to success: `0`
- Retry taxonomy: `none`
- Latest diagnosis dossier: `T73.md`
