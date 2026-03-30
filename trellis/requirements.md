# Trellis Agent Constitution

## Design Principles

1. **Autograd-compatible**: All numerical code must work with `autograd.numpy`. No in-place array mutation (`a[i] = x`), no `np.searchsorted`. Use functional patterns.
2. **Protocol-based**: Instruments implement `Instrument`, curves implement `DiscountCurve`, data sources implement `DataProvider`. New types should follow these protocols.
3. **Test-first**: Every new module must have tests. Run `pytest` before declaring success.
4. **Minimal dependencies**: Core functionality uses only numpy and autograd. External data providers are optional.
5. **Continuous compounding**: Internal rates are continuously compounded. Convert from market conventions (BEY, money market) at ingestion.

## Conventions

- Day count: ACT/ACT for Treasury bonds, ACT/360 for money market
- Year fractions: Use `trellis.core.date_utils.year_fraction()`
- Interpolation: Linear on zero rates, log-linear on discount factors
- Greeks: Compute via `autograd.grad` where possible; rate sensitivities prefer autodiff and fall back to finite differences when a payoff path is not differentiable

## Module Layout

- `core/` — types, protocols, autograd wrapper, date utilities
- `instruments/` — one file per instrument type
- `curves/` — yield curves and interpolation
- `data/` — market data providers and caching
- `engine/` — pricing orchestration and analytics
- `agent/` — AI agent infrastructure (introspection, tools, prompts, planning, building, execution)

## When Generating Code

- Place new instruments in `trellis/instruments/`
- Place new curves/models in `trellis/curves/`
- Always add corresponding tests in `tests/`
- Import `numpy` as `from trellis.core.differentiable import get_numpy; np = get_numpy()`
- Use type hints and docstrings
