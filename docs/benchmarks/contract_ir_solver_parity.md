# ContractIR Structural Compiler Parity

- Generated at: `2026-07-14T08:42:34.386044+00:00`
- Repo revision: `e8a9628521372df7629e6e47127aa0bdde8cc96a`

## Family Summary

| Family | Rep | Dec | Low | Parity | Prov | Exact authority | Phase 4 candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| asian_option | True | True | True | True | True | True | True |
| basket_option | True | True | True | True | True | True | True |
| digital_option | True | True | True | True | True | True | True |
| rate_style_swaption | True | True | True | True | True | True | True |
| vanilla_option | True | True | True | True | True | True | True |
| variance_swap | True | True | True | True | True | True | True |

## asian_option

- Arithmetic Asians now admit bounded analytical approximation and Monte Carlo structural lanes for European schedule-based equity-diffusion payoffs; broader family retirement remains governed by the admitted schedule-driven support contract.

| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| asian_call_analytical | request_decomposition | bound | compose_arithmetic_asian_analytical_call |  | True | True |
- `asian_call_analytical` value parity: structural=`635.2555626779798` reference=`635.2555626779798` abs_diff=`0.0`
| asian_put_analytical | request_decomposition | bound | compose_arithmetic_asian_analytical_put |  | True | True |
- `asian_put_analytical` value parity: structural=`0.38122467688113615` reference=`0.38122467688113615` abs_diff=`0.0`
| asian_call_monte_carlo | request_decomposition | bound | compose_arithmetic_asian_monte_carlo_call |  | True | True |
- `asian_call_monte_carlo` value parity: structural=`634.069513646976` reference=`632.5420227089422` abs_diff=`1.527490938033793`
| asian_put_monte_carlo | request_decomposition | bound | compose_arithmetic_asian_monte_carlo_put |  | True | True |
- `asian_put_monte_carlo` value parity: structural=`0.36505258985973554` reference=`0.2771686955877112` abs_diff=`0.08788389427202437`

## basket_option


| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| basket_call | request_decomposition | bound | helper_basket_option_call |  | True | True |
- `basket_call` value parity: structural=`5366.09274478414` reference=`5366.09274478414` abs_diff=`0.0`
| basket_put | request_decomposition | bound | helper_basket_option_put |  | True | True |
- `basket_put` value parity: structural=`1.5772591079770633e-05` reference=`1.5772591079770633e-05` abs_diff=`0.0`

## digital_option


| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| digital_cash_call | request_decomposition | bound | black76_cash_digital_call |  | True | True |
- `digital_cash_call` value parity: structural=`1.385924511208955` reference=`1.385924511208955` abs_diff=`0.0`
| digital_asset_put | request_decomposition | bound | black76_asset_digital_put |  | True | True |
- `digital_asset_put` value parity: structural=`37.69709044410678` reference=`37.69709044410678` abs_diff=`0.0`

## rate_style_swaption


| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| swaption_payer | semantic_blueprint | bound | helper_swaption_payer_black76 |  | True | True |
- `swaption_payer` value parity: structural=`3.782585433884869e-05` reference=`3.782585433884869e-05` abs_diff=`0.0`
| swaption_receiver | semantic_blueprint | bound | helper_swaption_receiver_black76 |  | True | True |
- `swaption_receiver` value parity: structural=`0.0899413945129767` reference=`0.0899413945129767` abs_diff=`0.0`

## vanilla_option


| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| vanilla_call | semantic_blueprint | bound | black76_vanilla_call |  | True | True |
- `vanilla_call` value parity: structural=`23.358571215221588` reference=`23.358571215221588` abs_diff=`0.0`
| vanilla_put | semantic_blueprint | bound | black76_vanilla_put |  | True | True |
- `vanilla_put` value parity: structural=`4.488815443711203` reference=`4.488815443711203` abs_diff=`0.0`

## variance_swap


| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| variance_swap | request_decomposition | bound | helper_equity_variance_swap |  | True | True |
- `variance_swap` value parity: structural=`221.11341162672662` reference=`221.11341162672662` abs_diff=`0.0`
