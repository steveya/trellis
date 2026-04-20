# ContractIR Structural Compiler Parity

- Generated at: `2026-04-20T04:31:31.628551+00:00`
- Repo revision: `d5213965728815ea1b4202d6dc8bbb6419717467`

## Family Summary

| Family | Rep | Dec | Low | Parity | Prov | Exact authority | Phase 4 candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| asian_option | True | True | False | False | False | False | False |
| basket_option | True | True | True | True | True | True | True |
| digital_option | True | True | True | True | True | True | True |
| rate_style_swaption | True | True | True | True | True | True | True |
| vanilla_option | True | True | True | True | True | True | True |
| variance_swap | True | True | True | True | True | True | True |

## asian_option

- Arithmetic Asians remain an explicit Phase 3 blocker: ContractIR decomposition exists, but the structural solver returns an intentional no-match until a checked arithmetic-Asian solver surface is admitted.

| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| asian_call_blocked | request_decomposition | no_match |  |  | None | True |
- `asian_call_blocked` blocker: `ContractIRSolverNoMatchError` — No admissible structural ContractIR solver declaration was found for method 'analytical' and outputs ('price',).

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
