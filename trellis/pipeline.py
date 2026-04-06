"""Pipeline: declarative batch pricing."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from uuid import uuid4

from trellis.book import Book, BookResult, ScenarioResultCube
from trellis.curves.scenario_packs import build_rate_curve_scenario_pack
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.session import Session


class BookExecutionPlan:
    """Reusable compiled scenario-batch plan for one book workflow."""

    def __init__(
        self,
        *,
        plan_id: str,
        book: Book,
        base_session: Session,
        scenario_specs: list[dict],
        measures: list[str],
        outputs: list[dict],
        data_source: str,
        as_of: date | str | None,
        discount_curve_name: str | None,
        vol_surface_name: str | None,
        greeks_mode: str,
    ):
        self._plan_id = str(plan_id).strip()
        self._book = book
        self._base_session = base_session
        self._scenario_specs = [deepcopy(spec) for spec in scenario_specs]
        self._measures = list(measures)
        self._outputs = [deepcopy(spec) for spec in outputs]
        self._data_source = str(data_source).strip()
        self._as_of = as_of
        self._discount_curve_name = discount_curve_name
        self._vol_surface_name = vol_surface_name
        self._greeks_mode = str(greeks_mode).strip()

    @property
    def base_session(self) -> Session:
        return self._base_session

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self._plan_id,
            "plan_type": "book_scenario_batch",
            "scenario_count": len(self._scenario_specs),
            "scenarios": [
                Pipeline._compute_plan_scenario_spec(spec)
                for spec in self._scenario_specs
            ],
            "measures": list(self._measures),
            "greeks_mode": self._greeks_mode,
            "data_source": self._data_source,
            "as_of": None if self._as_of is None else str(self._as_of),
            "discount_curve_name": self._discount_curve_name,
            "vol_surface_name": self._vol_surface_name,
            "outputs": [deepcopy(spec) for spec in self._outputs],
        }

    def execute(self) -> ScenarioResultCube:
        results: dict[str, BookResult] = {}
        scenario_specs: dict[str, dict] = {}
        scenario_provenance: dict[str, dict] = {}
        for spec in self._scenario_specs:
            name = spec.get("name", Pipeline._scenario_name(spec))
            if name in results:
                raise ValueError(
                    f"Duplicate scenario name {name!r} in pipeline run output."
                )
            session = Pipeline._apply_scenario(self._base_session, spec)
            projected = session._run_governed_request(
                book=self._book,
                request_type="price",
                measures=self._measures,
                metadata={"greeks_mode": self._greeks_mode},
                success_outcome="pipeline_priced",
                failure_outcome="pipeline_failed",
                default_message="Governed pipeline execution failed.",
                success_details={"scenario_name": name},
            )
            results[name] = projected
            public_spec = Pipeline._public_scenario_spec(spec)
            scenario_specs[name] = public_spec
            scenario_provenance[name] = {
                "pipeline_session_id": self._plan_id,
                "data_source": self._data_source,
                "as_of": None if self._as_of is None else str(self._as_of),
                "measures": list(self._measures),
                "greeks_mode": self._greeks_mode,
                "discount_curve_name": self._discount_curve_name,
                "vol_surface_name": self._vol_surface_name,
                "scenario_spec": deepcopy(Pipeline._compute_plan_scenario_spec(spec)),
                "expanded_from": (
                    None
                    if "_expanded_from" not in spec
                    else Pipeline._public_scenario_spec(spec["_expanded_from"])
                ),
            }
            for out in self._outputs:
                Pipeline._write_output(out, name, projected)

        return ScenarioResultCube(
            results,
            scenario_specs=scenario_specs,
            scenario_provenance=scenario_provenance,
            compute_plan=self.to_dict(),
            compute_plan_object=self,
        )


class Pipeline:
    """Declarative, chainable batch-pricing pipeline.

    Example::

        results = (
            Pipeline()
            .instruments_from_csv("portfolio.csv")
            .market_data(source="treasury_gov", as_of="2024-11-15")
            .compute(["price", "dv01", "duration"])
            .scenarios([{"shift_bps": 0}, {"shift_bps": +100}])
            .run()
        )
    """

    def __init__(self):
        """Initialize an empty pipeline with no instruments or scenarios yet."""
        self._session_id: str = f"pipeline_{uuid4().hex[:12]}"
        self._book: Book | None = None
        self._curve: YieldCurve | None = None
        self._market_snapshot: MarketSnapshot | None = None
        self._discount_curve_name: str | None = None
        self._vol_surface_name: str | None = None
        self._data_source: str = "treasury_gov"
        self._as_of: date | str | None = None
        self._measures: list[str] | None = None
        self._scenarios: list[dict] | None = None
        self._outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Builder methods (return self for chaining)
    # ------------------------------------------------------------------

    def instruments_from_csv(self, path: str) -> Pipeline:
        """Load a bond book from CSV and return the pipeline for chaining."""
        self._book = Book.from_csv(path)
        return self

    def instruments_from_dataframe(self, df) -> Pipeline:
        """Load a bond book from a pandas DataFrame and return ``self``."""
        self._book = Book.from_dataframe(df)
        return self

    def instruments(self, book: Book) -> Pipeline:
        """Attach an already-constructed book to the pipeline."""
        self._book = book
        return self

    def market_data(
        self,
        source: str = "treasury_gov",
        as_of: date | str | None = "latest",
        curve: YieldCurve | None = None,
        snapshot: MarketSnapshot | None = None,
        discount_curve: str | None = None,
        vol_surface_name: str | None = None,
    ) -> Pipeline:
        """Configure the base market data source or explicit snapshot.

        Use ``curve=`` for a single explicit discount curve or ``snapshot=``
        for a richer named market snapshot with discount, forecast, vol,
        credit, and FX components.
        """
        if snapshot is not None and curve is not None:
            raise ValueError("Pass either curve= or snapshot=, not both")
        self._data_source = source
        self._as_of = as_of
        self._curve = curve
        self._market_snapshot = snapshot
        self._discount_curve_name = discount_curve
        self._vol_surface_name = vol_surface_name
        return self

    def compute(self, measures: list[str]) -> Pipeline:
        """Choose the measures to compute for each scenario run."""
        self._measures = measures
        return self

    def scenarios(self, specs: list[dict]) -> Pipeline:
        """Register scenario specifications such as parallel shifts or tenor bumps."""
        self._scenarios = specs
        return self

    def output_parquet(self, path_template: str) -> Pipeline:
        """Write each scenario result to a parquet path template."""
        self._outputs.append({"format": "parquet", "path": path_template})
        return self

    def output_csv(self, path_template: str) -> Pipeline:
        """Write each scenario result to a CSV path template."""
        self._outputs.append({"format": "csv", "path": path_template})
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def compile_request(self):
        """Compile the pipeline input into the canonical platform request."""
        plan = self.compile_compute_plan()

        from trellis.agent.platform_requests import make_pipeline_request

        return make_pipeline_request(
            book=self._book,
            market_snapshot=plan.base_session.market_snapshot,
            settlement=plan.base_session.settlement,
            measures=self._measures or ["price"],
            metadata={
                "scenario_count": len(plan.to_dict()["scenarios"]),
                "data_source": self._data_source,
                "greeks_mode": self._pricing_greeks_mode(),
                "discount_curve_name": self._discount_curve_name,
                "vol_surface_name": self._vol_surface_name,
                "compute_plan": plan.to_dict(),
            },
        )

    def compile_compute_plan(self) -> BookExecutionPlan:
        """Compile the configured pipeline into a reusable scenario-batch plan."""
        if self._book is None:
            raise ValueError("No instruments configured — cannot compile compute plan")

        base = Session(
            curve=self._curve,
            market_snapshot=self._market_snapshot,
            discount_curve=self._discount_curve_name,
            vol_surface_name=self._vol_surface_name,
            settlement=None,
            as_of=self._as_of,
            data_source=self._data_source,
        )
        scenario_list = self._expanded_scenarios(base)
        return BookExecutionPlan(
            plan_id=self._session_id,
            book=self._book,
            base_session=base,
            scenario_specs=scenario_list,
            measures=list(self._measures or ["price"]),
            outputs=list(self._outputs),
            data_source=self._data_source,
            as_of=self._as_of,
            discount_curve_name=self._discount_curve_name,
            vol_surface_name=self._vol_surface_name,
            greeks_mode=self._pricing_greeks_mode(),
        )

    def to_execution_context(
        self,
        *,
        run_mode=None,
        provider_bindings=None,
        policy_bundle_id: str | None = None,
        allow_mock_data: bool | None = None,
        require_provider_disclosure: bool | None = None,
        default_output_mode: str = "result_only",
        default_audit_mode: str = "summary",
        requested_persistence: str = "ephemeral",
        requested_snapshot_policy: str = "prefer_bound_snapshot",
        metadata: dict | None = None,
    ):
        """Normalize the pipeline configuration into explicit governed runtime context."""
        from trellis.platform.context import execution_context_from_pipeline

        return execution_context_from_pipeline(
            self,
            run_mode=run_mode,
            provider_bindings=provider_bindings,
            policy_bundle_id=policy_bundle_id,
            allow_mock_data=allow_mock_data,
            require_provider_disclosure=require_provider_disclosure,
            default_output_mode=default_output_mode,
            default_audit_mode=default_audit_mode,
            requested_persistence=requested_persistence,
            requested_snapshot_policy=requested_snapshot_policy,
            metadata=metadata,
        )

    def run(self) -> ScenarioResultCube:
        """Execute the pipeline and return a dict-like scenario result cube."""
        if self._book is None:
            raise ValueError("No instruments configured — call instruments(), "
                             "instruments_from_csv(), or instruments_from_dataframe()")
        return self.compile_compute_plan().execute()

    def _pricing_greeks_mode(self) -> str:
        """Return the pricing greeks mode implied by configured pipeline measures."""
        if self._measures is None:
            return "all"
        return "none" if not [measure for measure in self._measures if measure != "price"] else "explicit"

    @staticmethod
    def _scenario_name(spec: dict) -> str:
        """Derive a readable scenario name from a scenario specification."""
        if "shift_bps" in spec:
            return f"shift_{Pipeline._format_bps_label(spec['shift_bps'], signed=True)}"
        if "tenor_bumps" in spec:
            return Pipeline._tenor_bump_name(spec["tenor_bumps"])
        if "quote_bucket_bumps" in spec:
            return Pipeline._quote_bucket_bump_name(spec["quote_bucket_bumps"])
        return "base"

    @staticmethod
    def _apply_scenario(base: Session, spec: dict) -> Session:
        """Apply a scenario spec to a base session and return the shocked session."""
        session = base
        if spec.get("methodology") == "curve_rebuild" or "quote_bucket_bumps" in spec:
            quote_bucket_bumps = dict(spec.get("quote_bucket_bumps") or {})
            if quote_bucket_bumps:
                session = session.with_bootstrap_quote_bumps(
                    quote_bucket_bumps,
                    curve_name=spec.get("selected_curve_name"),
                )
            else:
                raise ValueError(
                    "curve_rebuild scenarios require quote_bucket_bumps for replay."
                )
            return session
        if "shift_bps" in spec:
            session = session.with_curve_shift(spec["shift_bps"])
        if "tenor_bumps" in spec:
            session = session.with_tenor_bumps(spec["tenor_bumps"])
        return session

    def _expanded_scenarios(self, base: Session) -> list[dict]:
        """Expand named scenario packs into concrete scenario specifications."""
        scenario_list = self._scenarios or [{"name": "base"}]
        expanded: list[dict] = []
        for spec in scenario_list:
            if "scenario_template" in spec:
                template_name = str(spec["scenario_template"]).strip()
                template_spec = self._resolve_scenario_template(base, template_name)
                overrides = {
                    key: deepcopy(value)
                    for key, value in spec.items()
                    if key != "scenario_template"
                }
                for template_entry in self._template_scenario_specs(template_spec):
                    merged_spec = dict(template_entry)
                    merged_spec.update(overrides)
                    self._append_expanded_scenarios(
                        expanded,
                        base,
                        merged_spec,
                        expanded_from={"scenario_template": template_name},
                    )
                continue
            self._append_expanded_scenarios(expanded, base, spec)
        return expanded

    @staticmethod
    def _template_scenario_specs(template_spec: dict[str, object]) -> list[dict]:
        scenarios = template_spec.get("scenarios")
        if isinstance(scenarios, list):
            return [dict(item) for item in scenarios if isinstance(item, dict)]
        return [dict(template_spec)]

    def _resolve_scenario_template(self, base: Session, template_name: str) -> dict[str, object]:
        metadata = dict(getattr(base.market_snapshot, "metadata", {}) or {})
        templates = dict(metadata.get("scenario_templates") or {})
        template_spec = templates.get(template_name)
        if not isinstance(template_spec, dict):
            raise ValueError(f"Unknown saved scenario template: {template_name!r}")
        return dict(template_spec)

    def _append_expanded_scenarios(
        self,
        expanded: list[dict],
        base: Session,
        spec: dict,
        *,
        expanded_from: dict | None = None,
    ) -> None:
        if "scenario_pack" not in spec or Pipeline._is_concrete_scenario_spec(spec):
            projected = dict(spec)
            if expanded_from is not None:
                projected["_expanded_from"] = deepcopy(expanded_from)
            expanded.append(projected)
            return

        scenarios = build_rate_curve_scenario_pack(
            base.curve,
            pack=str(spec["scenario_pack"]),
            bucket_tenors=tuple(
                float(tenor)
                for tenor in spec.get("bucket_tenors", (2.0, 5.0, 10.0, 30.0))
            ),
            amplitude_bps=float(spec.get("amplitude_bps", 25.0)),
        )
        for scenario in scenarios:
            projected = scenario.to_pipeline_spec()
            projected["_expanded_from"] = (
                deepcopy(expanded_from)
                if expanded_from is not None
                else deepcopy(spec)
            )
            expanded.append(projected)

    @staticmethod
    def _public_scenario_spec(spec: dict) -> dict:
        """Project one scenario spec into the stable external cube shape."""
        public = {
            key: deepcopy(value)
            for key, value in spec.items()
            if not key.startswith("_")
        }
        public.setdefault("name", Pipeline._scenario_name(public))
        return public

    @staticmethod
    def _compute_plan_scenario_spec(spec: dict) -> dict:
        public = Pipeline._public_scenario_spec(spec)
        if "_expanded_from" in spec:
            public["expanded_from"] = Pipeline._public_scenario_spec(spec["_expanded_from"])
        return public

    @staticmethod
    def _is_concrete_scenario_spec(spec: dict) -> bool:
        return any(
            key in spec
            for key in ("shift_bps", "tenor_bumps", "quote_bucket_bumps")
        )

    @staticmethod
    def _format_numeric_label(value: object, *, signed: bool = False) -> str:
        numeric = float(value)
        normalized = f"{abs(numeric):g}".replace(".", "p")
        if numeric < 0.0:
            prefix = "m"
        elif numeric > 0.0 and signed:
            prefix = "p"
        else:
            prefix = ""
        return f"{prefix}{normalized}"

    @staticmethod
    def _format_bps_label(value: object, *, signed: bool = False) -> str:
        return f"{Pipeline._format_numeric_label(value, signed=signed)}bp"

    @staticmethod
    def _format_tenor_label(value: object) -> str:
        return f"{Pipeline._format_numeric_label(value)}y"

    @staticmethod
    def _tenor_bump_name(bumps: dict) -> str:
        entries = sorted(
            (
                float(tenor),
                float(bump),
            )
            for tenor, bump in dict(bumps or {}).items()
        )
        if not entries:
            return "tenor_bump"
        parts = [
            f"{Pipeline._format_tenor_label(tenor)}_{Pipeline._format_bps_label(bump, signed=True)}"
            for tenor, bump in entries
        ]
        return "tenor_bump_" + "_".join(parts)

    @staticmethod
    def _quote_bucket_bump_name(bumps: dict) -> str:
        entries = sorted(
            (str(bucket_id).strip(), float(bump))
            for bucket_id, bump in dict(bumps or {}).items()
        )
        if not entries:
            return "quote_bucket_bump"
        parts = [
            f"{bucket_id.lower()}_{Pipeline._format_bps_label(bump, signed=True)}"
            for bucket_id, bump in entries
        ]
        return "quote_bucket_bump_" + "_".join(parts)

    @staticmethod
    def _write_output(out: dict, scenario_name: str, br: BookResult):
        """Persist a scenario result using the configured output formatter."""
        path = out["path"].format(
            date=date.today().isoformat(),
            scenario=scenario_name,
        )
        df = br.to_dataframe()
        if out["format"] == "parquet":
            df.to_parquet(path, index=False)
        elif out["format"] == "csv":
            df.to_csv(path, index=False)
