# Trellis

Trellis is an AI-augmented pricing platform for quantitative finance.
Ask for a price in natural language, work directly from Python when you
need control, and drop down into sessions, payoffs, and numerical
methods when the workflow calls for it.

Trellis currently has three public faces:

- `trellis.ask(...)` and `Session.ask(...)` for agent-first pricing
- package-level Python APIs for reproducible library workflows
- `trellis-ui` as an experimental companion interface

## Installation

```bash
pip install trellis

# Optional runtime dependencies are installed separately today
pip install openai      # or: pip install anthropic
pip install requests fredapi
```

Install external comparison libraries such as QuantLib or FinancePy separately
when you need cross-validation coverage.

## Quick Examples

### Ask for a price

```python
import trellis

# Requires an installed provider client plus OPENAI_API_KEY or ANTHROPIC_API_KEY
result = trellis.ask("Price a 5Y SOFR cap at 4% on $10M")
print(result.price)
print(result.payoff_class)
print(result.matched_existing)
```

### Work offline with deterministic sample data

```python
import trellis

s = trellis.quickstart()
bond = trellis.sample_bond_10y()
result = s.price(bond)

print(result.clean_price)
print(result.greeks["dv01"])
```

## Documentation

- Start with [`docs/quickstart.rst`](docs/quickstart.rst)
- Follow the tutorials in [`docs/tutorials/index.rst`](docs/tutorials/index.rst)
- Use the high-level workflows in [`docs/user_guide`](docs/user_guide)
- Use [`docs/quant/index.rst`](docs/quant/index.rst) for pricing constructs, extension patterns, and knowledge maintenance
- Use [`docs/developer/index.rst`](docs/developer/index.rst) for hosting, agents, traces, and task/eval operations
- Treat experimental features accordingly:
  - agent-built payoffs
  - `trellis-ui`
  - live market-data auto-resolution beyond the mock provider

## Documentation Policy

The public docs are maintained around the package-level surface first.
Examples default to reproducible mock data unless a section is explicitly
marked as live-data or API-key dependent.
