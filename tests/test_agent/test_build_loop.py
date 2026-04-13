"""Tests for the agent build loop with mocked LLM responses."""

import math
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trellis.agent.planner import FieldDef, SpecSchema
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.fx import FXRate
from trellis.models.black import black76_call, garman_kohlhagen_call
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)

# A known-good complete module matching the static SwaptionSpec schema.
MOCK_MODULE_CODE = '''\
"""Agent-generated payoff: European payer swaption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.rate_style_swaption import (
    price_swaption_black76_raw,
    resolve_swaption_black76_inputs,
)


@dataclass(frozen=True)
class SwaptionSpec:
    """Specification for European payer swaption."""
    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class SwaptionPayoff:
    """European payer swaption."""

    def __init__(self, spec: SwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> SwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        resolved = resolve_swaption_black76_inputs(market_state, self._spec)
        return float(price_swaption_black76_raw(resolved))
'''

BAD_IMPORT_MODULE_CODE = MOCK_MODULE_CODE.replace(
    "from trellis.models.rate_style_swaption import (\n    price_swaption_black76_raw,\n    resolve_swaption_black76_inputs,\n)",
    "from trellis.models.not_a_real_module import (\n    price_swaption_black76_raw,\n    resolve_swaption_black76_inputs,\n)",
)

UNAPPROVED_IMPORT_MODULE_CODE = MOCK_MODULE_CODE.replace(
    "from trellis.models.rate_style_swaption import (\n    price_swaption_black76_raw,\n    resolve_swaption_black76_inputs,\n)",
    "from trellis.models.processes.heston import Heston",
)

GOOD_QUANTO_MODULE_CODE = '''\
"""Compatibility adapter for the quanto analytical payoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state


REQUIREMENTS = frozenset(
    {
        "black_vol_surface",
        "discount_curve",
        "forward_curve",
        "fx_rates",
        "model_parameters",
        "spot",
    }
)


@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for the single-name quanto analytical adapter."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class QuantoOptionAnalyticalPayoff:
    """Compatibility payoff that delegates through the semantic-facing helper."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_quanto_option_analytical_from_market_state(market_state, self._spec))
'''

UNAPPROVED_QUANTO_IMPORT_MODULE_CODE = GOOD_QUANTO_MODULE_CODE.replace(
    "from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state",
    "from trellis.models.processes.heston import Heston",
)

AMERICAN_SPEC_SCHEMA = SpecSchema(
    class_name="AmericanOptionPayoff",
    spec_name="AmericanPutEquitySpec",
    requirements=["discount", "black_vol"],
    fields=[
        FieldDef("spot", "float", "Current spot price"),
        FieldDef("strike", "float", "Option strike price"),
        FieldDef("expiry_date", "date", "Option expiry date"),
        FieldDef("option_type", "str", "Option type", '"put"'),
        FieldDef("exercise_style", "str", "Exercise style", '"american"'),
    ],
)

BAD_AMERICAN_SEMANTIC_MODULE_CODE = '''\
"""Agent-generated payoff: American put option on equity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class AmericanPutEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "put"
    exercise_style: str = "american"


class AmericanOptionPayoff:
    def __init__(self, spec: AmericanPutEquitySpec):
        self._spec = spec

    @property
    def spec(self) -> AmericanPutEquitySpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        import numpy as np
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.monte_carlo.schemes import LaguerreBasis
        from trellis.models.processes import GBM

        spec = self._spec
        T = (spec.expiry_date - market_state.settlement).days / 365.25
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=4096, n_steps=64, seed=42, method="lsm")

        def payoff_fn(paths):
            return np.maximum(spec.strike - paths, 0.0)

        basis = LaguerreBasis()
        _ = basis
        return float(engine.price(spec.spot, T, payoff_fn, discount_rate=r)["price"])
'''

GOOD_AMERICAN_SEMANTIC_MODULE_CODE = '''\
"""Agent-generated payoff: American put option on equity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState


@dataclass(frozen=True)
class AmericanPutEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "put"
    exercise_style: str = "american"


class AmericanOptionPayoff:
    def __init__(self, spec: AmericanPutEquitySpec):
        self._spec = spec

    @property
    def spec(self) -> AmericanPutEquitySpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.monte_carlo.lsm import longstaff_schwartz
        from trellis.models.monte_carlo.schemes import LaguerreBasis
        from trellis.models.processes.gbm import GBM

        spec = self._spec
        np = get_numpy()
        T = year_fraction(market_state.settlement, spec.expiry_date)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=4096, n_steps=64, seed=42, method="exact")
        paths = engine.simulate(spec.spot, T)
        dt = T / engine.n_steps
        exercise_dates = list(range(1, engine.n_steps + 1))
        basis = LaguerreBasis()

        def payoff_fn(spots):
            return np.maximum(spec.strike - spots, 0.0)

        return float(
            longstaff_schwartz(
                paths,
                exercise_dates,
                payoff_fn,
                discount_rate=r,
                dt=dt,
                basis_fn=basis,
            )
        )
'''


def test_reference_modules_include_quanto_resolution_helper():
    from types import SimpleNamespace

    from trellis.agent.executor import _reference_modules

    modules = _reference_modules(
        pricing_plan=SimpleNamespace(method="analytical"),
        instrument_type="quanto_option",
    )

    assert ("trellis.models.resolution.quanto", "Quanto input-resolution helpers") in modules
    assert ("trellis.models.analytical.quanto", "Quanto analytical route helpers") in modules
    assert ("trellis.models.monte_carlo.quanto", "Quanto Monte Carlo route helpers") in modules


def test_reference_modules_do_not_use_mutable_cds_agent_module_for_single_name_cds():
    from types import SimpleNamespace

    from trellis.agent.executor import _reference_modules

    modules = _reference_modules(
        pricing_plan=SimpleNamespace(method="monte_carlo"),
        instrument_type="credit_default_swap",
    )

    assert all(module != "trellis.instruments._agent.cds" for module, _ in modules)
    assert ("trellis.models.monte_carlo", "Monte Carlo package exports") not in modules
    assert ("trellis.instruments.barrier_option", "BarrierOptionPayoff (MC reference)") not in modules


def test_reference_modules_include_generic_rate_tree_surfaces_for_callable_bond():
    from types import SimpleNamespace

    from trellis.agent.executor import _reference_modules

    modules = _reference_modules(
        pricing_plan=SimpleNamespace(method="rate_tree"),
        instrument_type="callable_bond",
    )

    assert ("trellis.models.trees.lattice", "Generic/calibrated lattice builders") in modules
    assert ("trellis.models.trees.models", "Tree model registry for BDT/Hull-White selection") in modules
    assert ("trellis.models.trees.control", "Lattice exercise/control helpers") in modules


def test_reference_modules_use_concrete_transform_surfaces_for_fft_pricing():
    from types import SimpleNamespace

    from trellis.agent.executor import _reference_modules

    modules = _reference_modules(
        pricing_plan=SimpleNamespace(method="fft_pricing"),
        instrument_type="european_option",
    )

    assert ("trellis.models.transforms", "Transform package exports") not in modules
    assert ("trellis.models.transforms.fft_pricer", "FFT transform pricer") in modules
    assert ("trellis.models.transforms.cos_method", "COS transform pricer") in modules
    assert ("trellis.models.equity_option_transforms", "Vanilla equity transform helper") in modules


def test_reference_modules_use_concrete_copula_surfaces():
    from types import SimpleNamespace

    from trellis.agent.executor import _reference_modules

    modules = _reference_modules(
        pricing_plan=SimpleNamespace(method="copula"),
        instrument_type="nth_to_default",
    )

    assert ("trellis.models.copulas", "Copula package exports") not in modules
    assert ("trellis.models.copulas.factor", "Factor copula kernel") in modules
    assert ("trellis.models.copulas.gaussian", "Gaussian copula kernel") in modules
    assert ("trellis.models.copulas.student_t", "Student-t copula kernel") in modules
    assert ("trellis.models.credit_basket_copula", "Credit basket copula helper") in modules


def test_generate_quanto_monte_carlo_skeleton_uses_family_helper_surface():
    from types import SimpleNamespace

    from trellis.agent.executor import _generate_skeleton

    spec_schema = SpecSchema(
        class_name="QuantoOptionMonteCarloPayoff",
        spec_name="QuantoOptionSpec",
        requirements=[
            "discount_curve",
            "forward_curve",
            "black_vol_surface",
            "fx_rates",
            "spot",
            "model_parameters",
        ],
        fields=[
            FieldDef("notional", "float", "Notional"),
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry"),
            FieldDef("fx_pair", "str", "FX pair"),
            FieldDef("underlier_currency", "str", "Underlier currency", '"EUR"'),
            FieldDef("domestic_currency", "str", "Domestic currency", '"USD"'),
            FieldDef("option_type", "str", "Option type", '"call"'),
            FieldDef("quanto_correlation_key", "str | None", "Correlation key", "None"),
            FieldDef("day_count", "DayCountConvention", "Day count", "DayCountConvention.ACT_365"),
            FieldDef("n_paths", "int", "Path count", "50000"),
            FieldDef("n_steps", "int", "Step count", "252"),
        ],
    )

    skeleton = _generate_skeleton(
        spec_schema,
        "Quanto option: quanto-adjusted BS vs MC cross-currency",
        pricing_plan=SimpleNamespace(method="monte_carlo", model_to_build="quanto_option"),
        generation_plan=SimpleNamespace(method="monte_carlo", instrument_type="quanto_option"),
    )

    assert "from trellis.models.quanto_option import price_quanto_option_monte_carlo_from_market_state" in skeleton
    assert "return float(price_quanto_option_monte_carlo_from_market_state(market_state, spec))" in skeleton
def test_generate_module_reports_syntax_error_context(monkeypatch):
    from types import SimpleNamespace

    from trellis.agent.executor import _generate_module

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: "def broken(:\n    pass\n",
    )

    with pytest.raises(RuntimeError, match="SyntaxError"):
        _generate_module(
            skeleton="class Demo:\n    def evaluate(self, market_state):\n        raise NotImplementedError\n",
            spec_schema=SimpleNamespace(class_name="Demo", spec_name="DemoSpec", fields=[]),
            reference_sources={},
            model="test-model",
            max_retries=1,
        )


def test_generate_module_strips_fenced_python_and_compiles(monkeypatch):
    from trellis.agent.executor import _generate_module

    fenced_module = f"""\
```python
{MOCK_MODULE_CODE.rstrip()}
```
"""

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: fenced_module,
    )

    result = _generate_module(
        skeleton=MOCK_MODULE_CODE,
        spec_schema=SimpleNamespace(
            class_name="SwaptionPayoff",
            spec_name="SwaptionSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert result.source_report.fence_removed
    assert result.source_report.fence_language == "python"
    assert result.raw_code.startswith("```python")
    assert "class SwaptionPayoff" in result.code


def test_generate_module_recovers_partial_repair_fragment(monkeypatch):
    from trellis.agent.executor import _generate_module

    fragment = """from trellis.models.resolution.quanto import resolve_quanto_inputs

        spec = self._spec
        resolved = resolve_quanto_inputs(market_state, spec)
        return float(resolved["underlier_spot"])
"""

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: fragment,
    )

    result = _generate_module(
        skeleton="""from trellis.core.market_state import MarketState

class QuantoOptionSpec:
    pass


class QuantoOptionAnalyticalPayoff:
    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        raise NotImplementedError("evaluate not yet implemented")
""",
        spec_schema=SimpleNamespace(
            class_name="QuantoOptionAnalyticalPayoff",
            spec_name="QuantoOptionSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert "class QuantoOptionAnalyticalPayoff" in result.code
    assert "resolve_quanto_inputs" in result.code
    assert 'return float(resolved["underlier_spot"])' in result.code


def test_generate_module_recovers_compilable_evaluate_only_fragment(monkeypatch):
    from trellis.agent.executor import _generate_module

    fragment = """def evaluate(self, market_state: MarketState) -> float:
    spec = self._spec
    return float(spec.strike)
"""

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: fragment,
    )

    result = _generate_module(
        skeleton="""from trellis.core.market_state import MarketState

class SmokeSpec:
    strike: float


class SmokePayoff:
    def __init__(self, spec: SmokeSpec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        raise NotImplementedError("evaluate not yet implemented")
""",
        spec_schema=SimpleNamespace(
            class_name="SmokePayoff",
            spec_name="SmokeSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert "class SmokePayoff" in result.code
    assert "def evaluate(self, market_state: MarketState) -> float:" in result.code


def test_generate_module_recovers_imports_plus_evaluate_function_fragment(monkeypatch):
    from trellis.agent.executor import _generate_module

    fragment = """from trellis.models.resolution.quanto import resolve_quanto_inputs

def evaluate(self, market_state: MarketState) -> float:
    spec = self._spec
    resolved = resolve_quanto_inputs(market_state, spec)
    return float(resolved["underlier_spot"])
"""

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: fragment,
    )

    result = _generate_module(
        skeleton="""from trellis.core.market_state import MarketState

class QuantoOptionSpec:
    pass


class QuantoOptionAnalyticalPayoff:
    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        raise NotImplementedError("evaluate not yet implemented")
""",
        spec_schema=SimpleNamespace(
            class_name="QuantoOptionAnalyticalPayoff",
            spec_name="QuantoOptionSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert "class QuantoOptionAnalyticalPayoff" in result.code
    assert "def evaluate(self, market_state: MarketState) -> float:" in result.code
    assert "resolve_quanto_inputs(market_state, spec)" in result.code


def test_generate_module_recovers_fragment_with_offset_tail_indentation(monkeypatch):
    from trellis.agent.executor import _generate_module

    fragment = """spec = self._spec
        if market_state.discount is None:
            raise ValueError("missing discount")
        if market_state.credit_curve is None:
            raise ValueError("missing credit")
        spread = float(spec.spread)
        return spread
"""

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: fragment,
    )

    result = _generate_module(
        skeleton="""from trellis.core.market_state import MarketState

class SmokeSpec:
    spread: float


class SmokePayoff:
    def __init__(self, spec: SmokeSpec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        raise NotImplementedError("evaluate not yet implemented")
""",
        spec_schema=SimpleNamespace(
            class_name="SmokePayoff",
            spec_name="SmokeSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert "if market_state.credit_curve is None:" in result.code
    assert "return spread" in result.code


def test_generate_module_recovers_fragment_missing_indent_after_if(monkeypatch):
    from trellis.agent.executor import _generate_module

    fragment = """spread = float(spec.spread)
if spread > 1.0:
spread *= 1e-4
return spread
"""

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: fragment,
    )

    result = _generate_module(
        skeleton="""from trellis.core.market_state import MarketState

class SmokeSpec:
    spread: float


class SmokePayoff:
    def __init__(self, spec: SmokeSpec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        raise NotImplementedError("evaluate not yet implemented")
""",
        spec_schema=SimpleNamespace(
            class_name="SmokePayoff",
            spec_name="SmokeSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert "if spread > 1.0:" in result.code
    assert "spread *= 1e-4" in result.code


def test_generate_module_recovers_evaluate_body_from_malformed_full_module(monkeypatch):
    from trellis.agent.executor import _generate_module

    malformed_module = '''"""Agent-generated payoff: Quanto option."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state

    from importlib import import_module


@dataclass(frozen=True)
class QuantoOptionSpec:
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class QuantoOptionAnalyticalPayoff:
    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates", "model_parameters", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_quanto_option_analytical_from_market_state(market_state, self._spec))
'''

    monkeypatch.setattr(
        "trellis.agent.config.llm_generate",
        lambda prompt, model=None: malformed_module,
    )

    result = _generate_module(
        skeleton="""from trellis.core.market_state import MarketState
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state

class QuantoOptionSpec:
    pass


class QuantoOptionAnalyticalPayoff:
    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        # return float(price_quanto_option_analytical_from_market_state(market_state, spec))
        raise NotImplementedError("evaluate not yet implemented")
""",
        spec_schema=SimpleNamespace(
            class_name="QuantoOptionAnalyticalPayoff",
            spec_name="QuantoOptionSpec",
            fields=[],
        ),
        reference_sources={},
        model="test-model",
        max_retries=1,
    )

    compile(result.code, "<recovered>", "exec")
    assert "class QuantoOptionAnalyticalPayoff" in result.code
    assert "return float(price_quanto_option_analytical_from_market_state(market_state, spec))" in result.code


def test_validate_build_critic_path_uses_stage_helpers_without_nameerror(monkeypatch, caplog):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from trellis.agent.executor import _validate_build

    class DummyPayoff:
        @property
        def requirements(self) -> set[str]:
            return set()

        def evaluate(self, market_state) -> float:
            return 1.0

    monkeypatch.setattr(
        "trellis.agent.executor._make_test_payoff",
        lambda payoff_cls, spec_schema, settle: DummyPayoff(),
    )
    monkeypatch.setattr(
        "trellis.agent.review_policy.determine_review_policy",
        lambda **kwargs: SimpleNamespace(
            run_critic=True,
            risk_level="high",
            critic_reason="test",
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.validation_bundles.select_validation_bundle",
        lambda **kwargs: SimpleNamespace(bundle_id="demo", checks=(), categories={}),
    )
    monkeypatch.setattr(
        "trellis.agent.validation_bundles.execute_validation_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            failures=[],
            failure_details=(),
            executed_checks=(),
            skipped_checks=(),
        ),
    )

    captured = {}

    @contextmanager
    def fake_usage_stage(stage, metadata=None):
        captured["stage"] = stage
        captured["metadata"] = metadata or {}
        yield []

    monkeypatch.setattr("trellis.agent.config.get_model_for_stage", lambda stage, model=None: "critic-model")
    monkeypatch.setattr("trellis.agent.config.llm_usage_stage", fake_usage_stage)
    monkeypatch.setattr("trellis.agent.config.enforce_llm_token_budget", lambda stage=None: None)
    monkeypatch.setattr("trellis.agent.config.summarize_llm_usage", lambda usage: {})
    critic_call = {}

    def fake_critique(
        code,
        description,
        knowledge_context="",
        model=None,
        *,
        generation_plan=None,
        available_checks=None,
        json_max_retries=None,
        allow_text_fallback=True,
        text_max_retries=None,
    ):
        critic_call["available_checks"] = [check.check_id for check in available_checks or ()]
        critic_call["json_max_retries"] = json_max_retries
        critic_call["allow_text_fallback"] = allow_text_fallback
        critic_call["text_max_retries"] = text_max_retries
        return []

    monkeypatch.setattr("trellis.agent.critic.critique", fake_critique)
    monkeypatch.setattr(
        "trellis.agent.arbiter.run_critic_tests",
        lambda concerns, payoff, **kwargs: [],
    )

    caplog.set_level("WARNING")
    failures = _validate_build(
        payoff_cls=DummyPayoff,
        code="class Demo: pass",
        description="Demo analytical payoff",
        spec_schema=SimpleNamespace(class_name="DemoPayoff", spec_name="DemoSpec", fields=[]),
        validation="standard",
        model="gpt-5-mini",
        pricing_plan=SimpleNamespace(method="analytical", required_market_data=set()),
        product_ir=SimpleNamespace(instrument="european_option"),
        build_meta={},
        attempt_number=1,
    )

    assert failures == []
    assert captured["stage"] == "critic"
    assert captured["metadata"]["model"] == "critic-model"
    assert critic_call["available_checks"] == [
        "price_non_negative",
        "volatility_input_usage",
    ]
    assert critic_call["json_max_retries"] is None
    assert critic_call["allow_text_fallback"] is True
    assert "get_model_for_stage" not in caplog.text


def test_validate_build_passes_bounded_standard_critic_policy(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from trellis.agent.executor import _validate_build

    class DummyPayoff:
        @property
        def requirements(self) -> set[str]:
            return set()

        def evaluate(self, market_state) -> float:
            return 1.0

    monkeypatch.setattr(
        "trellis.agent.executor._make_test_payoff",
        lambda payoff_cls, spec_schema, settle: DummyPayoff(),
    )
    monkeypatch.setattr(
        "trellis.agent.review_policy.determine_review_policy",
        lambda **kwargs: SimpleNamespace(
            run_critic=True,
            risk_level="high",
            critic_reason="test",
            critic_mode="advisory",
            critic_json_max_retries=0,
            critic_allow_text_fallback=False,
            critic_text_max_retries=0,
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.validation_bundles.select_validation_bundle",
        lambda **kwargs: SimpleNamespace(bundle_id="demo", checks=(), categories={}),
    )
    monkeypatch.setattr(
        "trellis.agent.validation_bundles.execute_validation_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            failures=[],
            failure_details=(),
            executed_checks=(),
            skipped_checks=(),
        ),
    )

    @contextmanager
    def fake_usage_stage(stage, metadata=None):
        yield []

    monkeypatch.setattr("trellis.agent.config.get_model_for_stage", lambda stage, model=None: "critic-model")
    monkeypatch.setattr("trellis.agent.config.llm_usage_stage", fake_usage_stage)
    monkeypatch.setattr("trellis.agent.config.enforce_llm_token_budget", lambda stage=None: None)
    monkeypatch.setattr("trellis.agent.config.summarize_llm_usage", lambda usage: {})

    captured = {}

    def fake_critique(
        code,
        description,
        knowledge_context="",
        model=None,
        *,
        generation_plan=None,
        available_checks=None,
        json_max_retries=None,
        allow_text_fallback=True,
        text_max_retries=None,
    ):
        captured["available_checks"] = [check.check_id for check in available_checks or ()]
        captured["json_max_retries"] = json_max_retries
        captured["allow_text_fallback"] = allow_text_fallback
        captured["text_max_retries"] = text_max_retries
        return []

    monkeypatch.setattr("trellis.agent.critic.critique", fake_critique)
    monkeypatch.setattr(
        "trellis.agent.arbiter.run_critic_tests",
        lambda concerns, payoff, **kwargs: [],
    )

    failures = _validate_build(
        payoff_cls=DummyPayoff,
        code="class Demo: pass",
        description="Demo payoff",
        spec_schema=SimpleNamespace(class_name="DemoPayoff", spec_name="DemoSpec", fields=[]),
        validation="standard",
        model="gpt-5-mini",
        pricing_plan=SimpleNamespace(method="monte_carlo", required_market_data=set()),
        product_ir=SimpleNamespace(instrument="credit_default_swap"),
        build_meta={},
        attempt_number=1,
    )

    assert failures == []
    assert captured == {
        "available_checks": [],
        "json_max_retries": 0,
        "allow_text_fallback": False,
        "text_max_retries": 0,
    }


def test_actual_market_smoke_reports_runtime_error():
    from trellis.agent.executor import _smoke_test_actual_market_state

    class SmokeSpec:
        def __init__(self, strike, expiry_date):
            self.strike = strike
            self.expiry_date = expiry_date

    class SmokePayoff:
        def __init__(self, spec):
            self._spec = spec

        @property
        def requirements(self) -> set[str]:
            return set()

        def evaluate(self, market_state):
            raise TypeError("float() argument must be a string or a real number, not 'FXRate'")

    spec_schema = SimpleNamespace(
        spec_name="SmokeSpec",
        fields=[
            SimpleNamespace(name="strike", type="float"),
            SimpleNamespace(name="expiry_date", type="date"),
        ],
    )
    SmokeSpec.__module__ = __name__
    globals()["SmokeSpec"] = SmokeSpec
    try:
        failures = _smoke_test_actual_market_state(
            SmokePayoff,
            spec_schema,
            self_market_state:=TestBuildLoop()._quanto_market_state(),
        )
    finally:
        globals().pop("SmokeSpec", None)

    assert failures
    assert "actual market state smoke test failed" in failures[0].lower()
    assert "FXRate" in failures[0]


@patch("trellis.agent.executor._generate_module")
@patch("trellis.agent.executor._design_spec")
def test_build_does_not_return_last_failed_actual_market_candidate(
    mock_design_spec,
    mock_gen_mod,
):
    from trellis.agent.executor import build_payoff

    mock_design_spec.return_value = SpecSchema(
        class_name="SmokePayoff",
        spec_name="SmokeSpec",
        requirements=["discount_curve"],
        fields=[
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry date"),
        ],
    )
    mock_gen_mod.return_value = '''\
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState


@dataclass(frozen=True)
class SmokeSpec:
    strike: float
    expiry_date: date


class SmokePayoff:
    def __init__(self, spec: SmokeSpec):
        self._spec = spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        raise TypeError("synthetic smoke failure")
'''

    with pytest.raises(RuntimeError, match="Failed to build payoff after 2 attempts"):
        build_payoff(
            "Synthetic smoke payoff",
            {"discount_curve"},
            market_state=TestBuildLoop()._quanto_market_state(),
            instrument_type="european_option",
            force_rebuild=True,
            max_retries=2,
        )

GOOD_BERMUDAN_RATE_TREE_MODULE_CODE = '''\
"""Agent-generated payoff: Bermudan swaption on HW tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    notional: float
    strike: float
    exercise_dates: tuple[date, ...]
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class BermudanSwaptionPayoff:
    def __init__(self, spec: BermudanSwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BermudanSwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        from trellis.models.bermudan_swaption_tree import price_bermudan_swaption_tree

        return float(price_bermudan_swaption_tree(market_state, self._spec, model="hull_white"))
'''

BERMUDAN_SPEC_SCHEMA = SpecSchema(
    class_name="BermudanSwaptionPayoff",
    spec_name="BermudanSwaptionSpec",
    requirements=["black_vol", "discount", "forward_rate"],
    fields=[
        FieldDef("notional", "float", "Swaption notional"),
        FieldDef("strike", "float", "Fixed strike rate"),
        FieldDef("exercise_dates", "tuple[date, ...]", "Ordered Bermudan exercise dates"),
        FieldDef("swap_end", "date", "Underlying swap end date"),
        FieldDef("swap_frequency", "Frequency", "Swap payment frequency", "Frequency.SEMI_ANNUAL"),
        FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
        FieldDef("rate_index", "str | None", "Forecast curve key", "None"),
        FieldDef("is_payer", "bool", "Payer or receiver swaption", "True"),
    ],
)


class TestBuildLoop:

    def _fx_market_state(self) -> MarketState:
        dom = YieldCurve.flat(0.05)
        fgn = YieldCurve.flat(0.03)
        return MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=dom,
            forecast_curves={"EUR-DISC": fgn},
            fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
            spot=1.10,
            vol_surface=FlatVol(0.18),
        )

    def _quanto_market_state(self) -> MarketState:
        dom = YieldCurve.flat(0.05)
        fgn = YieldCurve.flat(0.03)
        return MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=dom,
            forecast_curves={"EUR-DISC": fgn},
            fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
            spot=100.0,
            underlier_spots={"EUR": 100.0},
            vol_surface=FlatVol(0.20),
            model_parameters={"quanto_correlation": 0.35},
        )

    @patch("trellis.agent.executor._generate_module")
    def test_build_payoff_with_mock(self, mock_gen_mod):
        """Full build loop with mocked LLM returning known-good module."""
        mock_gen_mod.return_value = MOCK_MODULE_CODE

        from trellis.agent.executor import build_payoff

        cls = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=True,
        )

        assert cls.__name__ == "SwaptionPayoff"
        assert hasattr(cls, "requirements")
        assert hasattr(cls, "evaluate")

    @patch("trellis.agent.executor._generate_module")
    def test_built_swaption_prices_correctly(self, mock_gen_mod):
        """The mock-built swaption produces a valid positive price."""
        mock_gen_mod.return_value = MOCK_MODULE_CODE

        from trellis.agent.executor import build_payoff

        SwaptionPayoff = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=True,
        )

        # Spec is deterministic — we know the exact field names
        mod = sys.modules["trellis.instruments._agent.swaption"]
        SwaptionSpec = mod.SwaptionSpec

        spec = SwaptionSpec(
            notional=1_000_000,
            strike=0.05,
            expiry_date=date(2025, 11, 15),
            swap_start=date(2025, 11, 15),
            swap_end=date(2030, 11, 15),
        )

        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            vol_surface=FlatVol(0.20),
        )

        pv = price_payoff(SwaptionPayoff(spec), ms)
        assert pv > 0
        assert pv < 1_000_000

    @patch("trellis.agent.executor._generate_module")
    def test_built_swaption_passes_invariants(self, mock_gen_mod):
        """The mock-built swaption passes the invariant suite."""
        mock_gen_mod.return_value = MOCK_MODULE_CODE

        from trellis.agent.executor import build_payoff
        from trellis.agent.invariants import run_invariant_suite

        SwaptionPayoff = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=True,
        )

        mod = sys.modules["trellis.instruments._agent.swaption"]
        SwaptionSpec = mod.SwaptionSpec

        spec = SwaptionSpec(
            notional=1_000_000, strike=0.05,
            expiry_date=date(2025, 11, 15),
            swap_start=date(2025, 11, 15),
            swap_end=date(2030, 11, 15),
        )

        def payoff_factory():
            return SwaptionPayoff(spec)

        def ms_factory(vol=0.20):
            return MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=YieldCurve.flat(0.05),
                vol_surface=FlatVol(vol),
            )

        passed, failures = run_invariant_suite(
            payoff_factory=payoff_factory,
            market_state_factory=ms_factory,
            is_option=True,
        )
        assert passed, f"Invariant failures: {failures}"

    @patch("trellis.agent.executor._generate_module")
    def test_build_reuses_deterministic_swaption_wrapper_without_codegen(self, mock_gen_mod):
        """Helper-backed swaption routes should bypass freeform code generation."""

        from trellis.agent.executor import build_payoff

        cls = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=True,
        )

        assert cls.__name__ == "SwaptionPayoff"
        mock_gen_mod.assert_not_called()

    @patch(
        "trellis.agent.executor._materialize_deterministic_exact_binding_module",
        return_value=None,
    )
    @patch("trellis.agent.executor._generate_module")
    def test_build_fails_on_unapproved_generated_module(self, mock_gen_mod, _mock_exact_binding):
        """Codegen still rejects unapproved Trellis imports when no exact binding wrapper applies."""
        mock_gen_mod.return_value = UNAPPROVED_QUANTO_IMPORT_MODULE_CODE

        from trellis.agent.executor import build_payoff

        with pytest.raises(RuntimeError, match="unapproved Trellis module"):
            build_payoff(
                "Quanto option: quanto-adjusted BS vs MC cross-currency",
                force_rebuild=True,
                fresh_build=True,
                validation="fast",
                instrument_type="quanto_option",
                preferred_method="analytical",
                max_retries=2,
            )

    @patch(
        "trellis.agent.executor._materialize_deterministic_exact_binding_module",
        return_value=None,
    )
    @patch("trellis.agent.executor._generate_module")
    def test_build_records_platform_failure_on_code_generation_error(
        self,
        mock_gen_mod,
        _mock_exact_binding,
    ):
        """Provider/code-generation failures should be recorded when the build falls back to codegen."""
        mock_gen_mod.side_effect = RuntimeError("OpenAI text request failed after 1 attempts")

        from trellis.agent import executor

        events: list[tuple[str, dict | None]] = []
        original_record_platform_event = executor._record_platform_event

        def _tracking_record_platform_event(compiled_request, event, **kwargs):
            events.append((event, kwargs.get("details")))
            return original_record_platform_event(compiled_request, event, **kwargs)

        with patch("trellis.agent.executor._record_platform_event", side_effect=_tracking_record_platform_event):
            with pytest.raises(RuntimeError, match="OpenAI text request failed"):
                executor.build_payoff(
                    "Quanto option: quanto-adjusted BS vs MC cross-currency",
                    force_rebuild=True,
                    fresh_build=True,
                    validation="fast",
                    instrument_type="quanto_option",
                    preferred_method="analytical",
                    max_retries=1,
                )

        failure_events = [details for event, details in events if event == "builder_attempt_failed"]
        assert failure_events
        assert failure_events[-1]["reason"] == "code_generation"
        assert failure_events[-1]["failure_count"] == 1

    @patch(
        "trellis.agent.executor._materialize_deterministic_exact_binding_module",
        return_value=None,
    )
    @patch("trellis.agent.executor._generate_module")
    def test_build_retries_after_code_generation_error(self, mock_gen_mod, _mock_exact_binding):
        """Code-generation failures should be retried before the build loop gives up."""
        mock_gen_mod.side_effect = [
            RuntimeError("OpenAI text request failed after 1 attempts"),
            GOOD_QUANTO_MODULE_CODE,
        ]

        from trellis.agent.executor import build_payoff

        cls = build_payoff(
            "Quanto option: quanto-adjusted BS vs MC cross-currency",
            force_rebuild=True,
            fresh_build=True,
            validation="fast",
            instrument_type="quanto_option",
            preferred_method="analytical",
            max_retries=2,
        )

        assert cls.__name__ == "QuantoOptionAnalyticalPayoff"
        assert mock_gen_mod.call_count == 2

    @patch("trellis.agent.executor._generate_module")
    def test_build_reuses_existing_fx_analytical_module(self, mock_gen_mod):
        """Vanilla FX analytical builds should reuse the deterministic adapter even on rebuilds."""
        from trellis.agent.executor import build_payoff

        payoff_cls = build_payoff(
            "FX option (EURUSD): GK analytical vs MC",
            force_rebuild=True,
            validation="fast",
            instrument_type="european_option",
            preferred_method="analytical",
        )

        mod = sys.modules[payoff_cls.__module__]
        spec_cls = mod.FXVanillaOptionSpec
        spec = spec_cls(
            notional=1_000_000,
            strike=1.08,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
            foreign_discount_key="EUR-DISC",
        )
        market_state = self._fx_market_state()
        pv = price_payoff(payoff_cls(spec), market_state)
        T = year_fraction(SETTLE, spec.expiry_date, spec.day_count)
        expected = 1_000_000 * garman_kohlhagen_call(
            1.10,
            1.08,
            0.18,
            T,
            market_state.discount.discount(T),
            market_state.forecast_curves["EUR-DISC"].discount(T),
        )

        mock_gen_mod.assert_not_called()
        assert payoff_cls.__name__ == "FXVanillaAnalyticalPayoff"
        assert pv == pytest.approx(expected, rel=1e-10)

    @patch("trellis.agent.executor._generate_module")
    def test_build_reuses_existing_fx_monte_carlo_module(self, mock_gen_mod):
        """Vanilla FX Monte Carlo builds should reuse the deterministic adapter even on rebuilds."""
        from trellis.agent.executor import build_payoff

        payoff_cls = build_payoff(
            "FX option (EURUSD): GK analytical vs MC",
            force_rebuild=True,
            validation="fast",
            instrument_type="european_option",
            preferred_method="monte_carlo",
        )

        mod = sys.modules[payoff_cls.__module__]
        spec_cls = mod.FXVanillaOptionSpec
        spec = spec_cls(
            notional=250_000,
            strike=1.08,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
            foreign_discount_key="EUR-DISC",
            n_paths=15000,
            n_steps=96,
        )
        market_state = self._fx_market_state()
        pv = price_payoff(payoff_cls(spec), market_state)
        T = year_fraction(SETTLE, spec.expiry_date, spec.day_count)
        expected = 250_000 * garman_kohlhagen_call(
            1.10,
            1.08,
            0.18,
            T,
            market_state.discount.discount(T),
            market_state.forecast_curves["EUR-DISC"].discount(T),
        )

        mock_gen_mod.assert_not_called()
        assert payoff_cls.__name__ == "FXVanillaMonteCarloPayoff"
        assert pv == pytest.approx(expected, rel=0.06)

    @patch("trellis.agent.executor._generate_module")
    def test_build_reuses_existing_quanto_analytical_module(self, mock_gen_mod):
        """Quanto analytical builds should reuse the checked-in deterministic adapter."""
        from trellis.agent.executor import build_payoff

        payoff_cls = build_payoff(
            "Quanto option: quanto-adjusted BS vs MC cross-currency",
            force_rebuild=True,
            validation="fast",
            instrument_type="quanto_option",
            preferred_method="analytical",
        )

        mod = sys.modules[payoff_cls.__module__]
        spec_cls = mod.QuantoOptionSpec
        spec = spec_cls(
            notional=250_000,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
            underlier_currency="EUR",
            domestic_currency="USD",
        )
        market_state = self._quanto_market_state()
        pv = price_payoff(payoff_cls(spec), market_state)
        T = year_fraction(SETTLE, spec.expiry_date, spec.day_count)
        domestic_df = market_state.discount.discount(T)
        foreign_df = market_state.forecast_curves["EUR-DISC"].discount(T)
        sigma_underlier = market_state.vol_surface.black_vol(T, spec.strike)
        sigma_fx = market_state.vol_surface.black_vol(T, market_state.fx_rates["EURUSD"].spot)
        quanto_forward = (
            market_state.underlier_spots["EUR"]
            * foreign_df
            / domestic_df
            * math.exp(
                -market_state.model_parameters["quanto_correlation"] * sigma_underlier * sigma_fx * T
            )
        )
        expected = 250_000 * domestic_df * black76_call(
            quanto_forward,
            spec.strike,
            sigma_underlier,
            T,
        )

        mock_gen_mod.assert_not_called()
        assert payoff_cls.__name__ == "QuantoOptionAnalyticalPayoff"
        assert pv == pytest.approx(expected, rel=1e-10)

    @patch(
        "trellis.agent.executor._materialize_deterministic_exact_binding_module",
        return_value=None,
    )
    @patch("trellis.agent.executor._generate_module")
    def test_build_fresh_build_can_fall_back_to_codegen_without_exact_binding(
        self,
        mock_gen_mod,
        _mock_exact_binding,
    ):
        """Fresh-build mode should still invoke codegen when exact binding materialization is unavailable."""
        from trellis.agent.executor import build_payoff

        mock_gen_mod.side_effect = RuntimeError("fresh-build invoked")

        with pytest.raises(RuntimeError, match="fresh-build invoked"):
            build_payoff(
                "Quanto option: quanto-adjusted BS vs MC cross-currency",
                force_rebuild=True,
                fresh_build=True,
                validation="fast",
                instrument_type="quanto_option",
                preferred_method="analytical",
            )

        assert mock_gen_mod.call_count == 3

    @patch("trellis.agent.builder.dynamic_import")
    @patch("trellis.agent.executor.write_module")
    @patch("trellis.agent.executor._generate_module")
    def test_build_fresh_build_writes_quanto_candidate_to_scratch_module(
        self,
        mock_gen_mod,
        mock_write_module,
        mock_dynamic_import,
    ):
        """Fresh-build candidates must not overwrite the checked-in deterministic route."""
        from types import ModuleType
        from types import SimpleNamespace

        from trellis.agent.executor import build_payoff

        mock_gen_mod.return_value = '''\
"""Compatibility adapter for the quanto analytical payoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state


REQUIREMENTS = frozenset(
    {
        "black_vol_surface",
        "discount_curve",
        "forward_curve",
        "fx_rates",
        "model_parameters",
        "spot",
    }
)


@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for the single-name quanto analytical adapter."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class QuantoOptionAnalyticalPayoff:
    """Compatibility payoff that delegates through the semantic-facing helper."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_quanto_option_analytical_from_market_state(market_state, self._spec))
'''
        scratch_path = Path("/tmp/quantooptionanalytical_fresh.py")
        mock_write_module.return_value = scratch_path
        mod = ModuleType("trellis.instruments._agent._fresh.quantooptionanalytical")
        setattr(mod, "QuantoOptionAnalyticalPayoff", type("QuantoOptionAnalyticalPayoff", (), {}))
        mock_dynamic_import.return_value = mod
        with patch(
            "trellis.agent.executor.validate_semantics",
            return_value=SimpleNamespace(ok=True, errors=()),
        ), patch(
            "trellis.agent.lite_review.review_generated_code",
            return_value=SimpleNamespace(ok=True, errors=(), issues=[]),
        ):
            build_payoff(
                "Quanto option: quanto-adjusted BS vs MC cross-currency",
                force_rebuild=True,
                fresh_build=True,
                validation="fast",
                instrument_type="quanto_option",
                preferred_method="analytical",
            )

        write_path = mock_write_module.call_args.args[0]
        assert write_path == "instruments/_agent/_fresh/quantooptionanalytical.py"

    @patch("trellis.agent.executor._generate_module")
    def test_build_reuses_existing_quanto_monte_carlo_module(self, mock_gen_mod):
        """Quanto Monte Carlo builds should reuse the checked-in deterministic adapter."""
        from trellis.agent.executor import build_payoff

        payoff_cls = build_payoff(
            "Quanto option: quanto-adjusted BS vs MC cross-currency",
            force_rebuild=True,
            validation="fast",
            instrument_type="quanto_option",
            preferred_method="monte_carlo",
        )

        mod = sys.modules[payoff_cls.__module__]
        spec_cls = mod.QuantoOptionSpec
        spec = spec_cls(
            notional=100_000,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
            underlier_currency="EUR",
            domestic_currency="USD",
            n_paths=20000,
            n_steps=128,
        )
        market_state = self._quanto_market_state()
        pv = price_payoff(payoff_cls(spec), market_state)
        T = year_fraction(SETTLE, spec.expiry_date, spec.day_count)
        domestic_df = market_state.discount.discount(T)
        foreign_df = market_state.forecast_curves["EUR-DISC"].discount(T)
        sigma_underlier = market_state.vol_surface.black_vol(T, spec.strike)
        sigma_fx = market_state.vol_surface.black_vol(T, market_state.fx_rates["EURUSD"].spot)
        quanto_forward = (
            market_state.underlier_spots["EUR"]
            * foreign_df
            / domestic_df
            * math.exp(
                -market_state.model_parameters["quanto_correlation"] * sigma_underlier * sigma_fx * T
            )
        )
        expected = 100_000 * domestic_df * black76_call(
            quanto_forward,
            spec.strike,
            sigma_underlier,
            T,
        )

        mock_gen_mod.assert_not_called()
        assert payoff_cls.__name__ == "QuantoOptionMonteCarloPayoff"
        assert pv == pytest.approx(expected, rel=0.08)

    @patch("trellis.agent.executor._generate_module")
    @patch("trellis.agent.executor._design_spec")
    def test_build_retries_after_semantic_validation_failures(
        self,
        mock_design_spec,
        mock_gen_mod,
        tmp_path,
    ):
        """Semantically invalid code is rejected before module write and retried."""
        mock_design_spec.return_value = AMERICAN_SPEC_SCHEMA
        mock_gen_mod.side_effect = [
            BAD_AMERICAN_SEMANTIC_MODULE_CODE,
            GOOD_AMERICAN_SEMANTIC_MODULE_CODE,
        ]

        from trellis.agent.executor import build_payoff

        target_path = tmp_path / "american_option_temp.py"

        def _write_to_temp(_module_path, content):
            target_path.write_text(content)
            return target_path

        with patch(
            "trellis.agent.executor.write_module",
            side_effect=_write_to_temp,
        ) as mock_write_module:
            cls = build_payoff(
                "American put option on equity",
                {"discount_curve", "black_vol_surface"},
                instrument_type="american_option",
                force_rebuild=True,
                validation="fast",
                max_retries=3,
            )

        assert cls.__name__ == "AmericanOptionPayoff"
        assert mock_gen_mod.call_count == 2
        assert mock_write_module.call_count == 1

    @patch("trellis.agent.executor._generate_module")
    @patch("trellis.agent.executor._design_spec")
    def test_rebuilt_american_payoff_prices_plausibly(
        self,
        mock_design_spec,
        mock_gen_mod,
        tmp_path,
    ):
        """A rebuilt American payoff follows the LSM route and produces a sane price."""
        mock_design_spec.return_value = AMERICAN_SPEC_SCHEMA
        mock_gen_mod.return_value = GOOD_AMERICAN_SEMANTIC_MODULE_CODE

        from trellis.agent.executor import build_payoff

        target_path = tmp_path / "american_option_temp.py"

        def _write_to_temp(_module_path, content):
            target_path.write_text(content)
            return target_path

        with patch(
            "trellis.agent.executor.write_module",
            side_effect=_write_to_temp,
        ):
            payoff_cls = build_payoff(
                "American put option on equity",
                {"discount_curve", "black_vol_surface"},
                instrument_type="american_option",
                force_rebuild=True,
                validation="fast",
                max_retries=1,
            )

        mod = sys.modules[payoff_cls.__module__]
        spec = mod.AmericanPutEquitySpec(
            spot=100.0,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            vol_surface=FlatVol(0.20),
        )

        pv = price_payoff(payoff_cls(spec), market_state)
        assert pv > 5.0
        assert pv < 8.0

    @patch("trellis.agent.executor._generate_module")
    def test_build_aborts_when_primitive_plan_has_blockers(self, mock_gen_mod):
        """Unsupported routes fail before code generation when primitive blockers are known."""
        from trellis.agent.executor import build_payoff

        with pytest.raises(
            RuntimeError,
            match="primitive planning blockers|Build gate blocked pre-generation",
        ):
            build_payoff(
                "American Asian barrier option under Heston with early exercise",
                {"discount_curve", "black_vol_surface"},
                force_rebuild=True,
                validation="fast",
                max_retries=1,
            )

        mock_gen_mod.assert_not_called()

    @patch("trellis.agent.executor._generate_module")
    @patch("trellis.agent.executor._design_spec")
    def test_rebuilt_bermudan_rate_tree_payoff_prices_plausibly(
        self,
        mock_design_spec,
        mock_gen_mod,
        tmp_path,
    ):
        """A rebuilt Bermudan swaption follows the rate-tree route and prices near the task reference."""
        mock_design_spec.return_value = BERMUDAN_SPEC_SCHEMA
        mock_gen_mod.return_value = GOOD_BERMUDAN_RATE_TREE_MODULE_CODE

        from trellis.agent.executor import build_payoff

        target_path = tmp_path / "bermudan_swaption_temp.py"

        def _write_to_temp(_module_path, content):
            target_path.write_text(content)
            return target_path

        with patch(
            "trellis.agent.executor.write_module",
            side_effect=_write_to_temp,
        ):
            payoff_cls = build_payoff(
                "Bermudan swaption: tree vs LSM MC",
                {"discount_curve", "forward_curve", "black_vol_surface"},
                instrument_type="bermudan_swaption",
                force_rebuild=True,
                validation="fast",
                max_retries=1,
            )

        mod = sys.modules[payoff_cls.__module__]
        spec = mod.BermudanSwaptionSpec(
            notional=100.0,
            strike=0.05,
            exercise_dates="2025-11-15,2026-11-15,2027-11-15,2028-11-15,2029-11-15",
            swap_end=date(2030, 11, 15),
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05, max_tenor=31.0),
            vol_surface=FlatVol(0.20),
        )
        pv = price_payoff(payoff_cls(spec), market_state)
        assert pv > 1.0
        assert pv < 4.0
