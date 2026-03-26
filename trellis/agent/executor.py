"""Top-level agent executor: plan → build → fetch → price → return."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace

from trellis.agent.codegen_guardrails import (
    build_generation_plan,
    validate_generated_imports,
)
from trellis.agent.blocker_planning import render_blocker_report
from trellis.agent.introspection import (
    find_symbol,
    get_package_tree,
    list_module_exports,
    read_module_source,
    search_lessons,
    search_package,
    search_tests,
)
from trellis.agent.prompts import system_prompt
from trellis.agent.semantic_validation import validate_semantics
from trellis.agent.tools import TOOLS
from trellis.agent.builder import write_module, run_tests
from trellis.agent.knowledge.import_registry import resolve_import_candidates


# Sentinel in the skeleton that gets replaced by the LLM-generated body.
EVALUATE_SENTINEL = '        raise NotImplementedError("evaluate not yet implemented")'

_DETERMINISTIC_SUPPORTED_ROUTE_MODULES = frozenset({
    "instruments/_agent/fxvanillaanalytical.py",
    "instruments/_agent/fxvanillamontecarlo.py",
})


def _handle_tool_call(name: str, input_data: dict) -> str:
    """Dispatch a tool call from the LLM agent."""
    if name == "inspect_library":
        tree = get_package_tree()
        return json.dumps(tree, indent=2, default=str)

    elif name == "read_module":
        try:
            src = read_module_source(input_data["module_path"])
            return src
        except Exception as e:
            return f"Error reading module: {e}"

    elif name == "find_symbol":
        try:
            matches = find_symbol(input_data["symbol"])
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error finding symbol: {e}"

    elif name == "list_exports":
        try:
            exports = list_module_exports(input_data["module_path"])
            return json.dumps(exports, indent=2)
        except Exception as e:
            return f"Error listing exports: {e}"

    elif name == "resolve_import_candidates":
        try:
            candidates = resolve_import_candidates(input_data["symbols"])
            return json.dumps(candidates, indent=2)
        except Exception as e:
            return f"Error resolving imports: {e}"

    elif name == "lookup_primitive_route":
        try:
            from trellis.agent.assembly_tools import lookup_primitive_route

            payload = lookup_primitive_route(
                description=input_data["description"],
                instrument_type=input_data.get("instrument_type"),
                preferred_method=input_data.get("preferred_method"),
            )
            return json.dumps(asdict(payload), indent=2)
        except Exception as e:
            return f"Error building primitive route plan: {e}"

    elif name == "build_thin_adapter_plan":
        try:
            from trellis.agent.assembly_tools import build_thin_adapter_plan
            from trellis.agent.codegen_guardrails import build_generation_plan
            from trellis.agent.knowledge.decompose import decompose_to_ir
            from trellis.agent.quant import (
                select_pricing_method,
                select_pricing_method_for_product_ir,
            )

            description = input_data["description"]
            instrument_type = input_data.get("instrument_type")
            preferred_method = input_data.get("preferred_method")
            class_name = input_data.get("class_name", "GeneratedPayoff")

            product_ir = decompose_to_ir(description, instrument_type=instrument_type)
            if preferred_method:
                pricing_plan = select_pricing_method_for_product_ir(
                    product_ir,
                    preferred_method=preferred_method,
                    context_description=description,
                )
            else:
                pricing_plan = select_pricing_method(description, instrument_type=instrument_type)
            generation_plan = build_generation_plan(
                pricing_plan=pricing_plan,
                instrument_type=instrument_type or getattr(product_ir, "instrument", None),
                inspected_modules=tuple(pricing_plan.method_modules),
                product_ir=product_ir,
            )
            spec_schema = SimpleNamespace(class_name=class_name, fields=[])
            payload = build_thin_adapter_plan(
                spec_schema,
                pricing_plan=pricing_plan,
                generation_plan=generation_plan,
            )
            return json.dumps(asdict(payload), indent=2)
        except Exception as e:
            return f"Error building thin adapter plan: {e}"

    elif name == "select_invariant_pack":
        try:
            from trellis.agent.assembly_tools import select_invariant_pack

            payload = select_invariant_pack(
                instrument_type=input_data.get("instrument_type"),
                method=input_data["method"],
            )
            return json.dumps(asdict(payload), indent=2)
        except Exception as e:
            return f"Error selecting invariant pack: {e}"

    elif name == "build_comparison_harness":
        try:
            from trellis.agent.assembly_tools import build_comparison_harness_plan

            payload = build_comparison_harness_plan(input_data["task"])
            return json.dumps(
                {
                    "targets": [asdict(target) for target in payload.targets],
                    "reference_target": payload.reference_target,
                    "tolerance_pct": payload.tolerance_pct,
                },
                indent=2,
            )
        except Exception as e:
            return f"Error building comparison harness plan: {e}"

    elif name == "capture_cookbook_candidate":
        try:
            from trellis.agent.assembly_tools import build_cookbook_candidate_payload

            payload = build_cookbook_candidate_payload(
                method=input_data["method"],
                description=input_data["description"],
                code=input_data["code"],
            )
            return json.dumps(payload or {}, indent=2)
        except Exception as e:
            return f"Error extracting cookbook candidate: {e}"

    elif name == "search_repo":
        try:
            matches = search_package(
                input_data["pattern"],
                limit=int(input_data.get("limit", 20)),
            )
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error searching repo: {e}"

    elif name == "search_tests":
        try:
            matches = search_tests(
                input_data["pattern"],
                limit=int(input_data.get("limit", 20)),
            )
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error searching tests: {e}"

    elif name == "search_lessons":
        try:
            matches = search_lessons(
                input_data["pattern"],
                limit=int(input_data.get("limit", 20)),
            )
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error searching lessons: {e}"

    elif name == "write_module":
        path = write_module(input_data["file_path"], input_data["content"])
        return f"Module written to {path}"

    elif name == "run_tests":
        result = run_tests(input_data.get("test_path"))
        return json.dumps(result, indent=2)

    elif name == "fetch_market_data":
        source = input_data.get("source", "fred")
        as_of_str = input_data.get("as_of")
        as_of = datetime.strptime(as_of_str, "%Y-%m-%d").date() if as_of_str else None

        if source == "fred":
            from trellis.data.fred import FredDataProvider
            data = FredDataProvider().fetch_yields(as_of)
        else:
            from trellis.data.treasury_gov import TreasuryGovDataProvider
            data = TreasuryGovDataProvider().fetch_yields(as_of)
        return json.dumps(data, indent=2)

    elif name == "execute_pricing":
        from trellis.instruments.bond import Bond
        from trellis.curves.yield_curve import YieldCurve
        from trellis.engine.pricer import price_instrument

        params = input_data["params"]
        curve_data = {float(k): float(v) for k, v in input_data["curve_data"].items()}
        curve = YieldCurve.from_treasury_yields(curve_data)

        instrument_type = input_data.get("instrument_type", "Bond")
        if instrument_type == "Bond":
            if "maturity_date" in params and isinstance(params["maturity_date"], str):
                params["maturity_date"] = datetime.strptime(params["maturity_date"], "%Y-%m-%d").date()
            bond = Bond(**params)
        else:
            return f"Unknown instrument type: {instrument_type}"

        result = price_instrument(bond, curve)
        return json.dumps({
            "clean_price": result.clean_price,
            "dirty_price": result.dirty_price,
            "accrued_interest": result.accrued_interest,
            "greeks": result.greeks,
        }, indent=2, default=str)

    return f"Unknown tool: {name}"


def execute(query: str, max_turns: int = 10, model: str = "claude-sonnet-4-6") -> dict:
    """Run the full agent loop for a natural-language pricing request."""
    from trellis.agent.config import get_anthropic_client

    client = get_anthropic_client()
    messages = [{"role": "user", "content": query}]
    sys_prompt = system_prompt()

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=sys_prompt,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [b for b in assistant_content if b.type == "tool_use"]
        if not tool_uses:
            text_parts = [b.text for b in assistant_content if hasattr(b, "text")]
            return {"response": "\n".join(text_parts), "turns": turn + 1}

        tool_results = []
        for tool_use in tool_uses:
            result = _handle_tool_call(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return {"response": "Max turns reached", "turns": max_turns}


def _record_platform_event(
    compiled_request,
    event: str,
    *,
    status: str = "info",
    success: bool | None = None,
    outcome: str | None = None,
    details: dict | None = None,
) -> None:
    """Best-effort request audit event emission."""
    if compiled_request is None:
        return
    try:
        from trellis.agent.platform_traces import append_platform_trace_event

        append_platform_trace_event(
            compiled_request,
            event,
            status=status,
            success=success,
            outcome=outcome,
            details=details,
        )
    except Exception:
        pass


def _append_agent_observation(
    build_meta: dict | None,
    agent: str,
    kind: str,
    summary: str,
    *,
    severity: str = "info",
    attempt: int | None = None,
    details=None,
) -> None:
    """Best-effort structured memory capture for cross-agent learning."""
    if build_meta is None:
        return
    observation = {
        "agent": agent,
        "kind": kind,
        "summary": summary,
        "severity": severity,
    }
    if attempt is not None:
        observation["attempt"] = attempt
    if details is not None:
        observation["details"] = details
    build_meta.setdefault("agent_observations", []).append(observation)


def _finalizes_in_executor(compiled_request) -> bool:
    """Standalone build requests terminate inside the executor."""
    if compiled_request is None:
        return False
    return getattr(compiled_request.request, "request_type", None) == "build"


# ---------------------------------------------------------------------------
# Structured payoff builder (two-step pipeline)
# ---------------------------------------------------------------------------

def build_payoff(
    payoff_description: str,
    requirements: set[str] | None = None,
    model: str | None = None,
    max_retries: int = 3,
    force_rebuild: bool = False,
    validation: str = "standard",
    market_state=None,
    instrument_type: str | None = None,
    preferred_method: str | None = None,
    request_metadata: Mapping[str, object] | None = None,
    compiled_request=None,
    build_meta: dict | None = None,
) -> type:
    """Build a Payoff class via the multi-agent pipeline.

    Pipeline:
    1. **Quant agent** selects the pricing method and data requirements
    2. **Data check** verifies required market data is available
    3. **Planner** determines spec schema and module path
    4. **Builder agent** generates the code using the prescribed method
    5. **Critic agent** reviews the code
    6. **Arbiter** validates with invariants
    """
    from trellis.agent.planner import plan_build
    from trellis.agent.quant import (
        check_data_availability,
        select_pricing_method,
        select_pricing_method_for_product_ir,
    )
    from trellis.agent.builder import dynamic_import, ensure_agent_package
    from trellis.agent.config import (
        enforce_llm_token_budget,
        get_default_model,
        get_model_for_stage,
        llm_usage_stage,
        summarize_llm_usage,
    )
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.platform_requests import compile_build_request
    from trellis.core.market_state import MissingCapabilityError

    model = model or get_default_model()
    product_ir = None
    if compiled_request is None:
        try:
            compiled_request = compile_build_request(
                payoff_description,
                instrument_type=instrument_type,
                market_snapshot=getattr(market_state, "market_snapshot", None),
                settlement=getattr(market_state, "settlement", None),
                model=model,
                preferred_method=preferred_method,
                metadata=request_metadata,
            )
            product_ir = compiled_request.product_ir
        except Exception:
            try:
                product_ir = decompose_to_ir(
                    payoff_description,
                    instrument_type=instrument_type,
                )
            except Exception:
                product_ir = None
    else:
        product_ir = compiled_request.product_ir

    if build_meta is not None and compiled_request is not None:
        build_meta.setdefault(
            "knowledge_summary",
            dict(getattr(compiled_request, "knowledge_summary", {}) or {}),
        )

    # Step 1: Quant agent selects method + data requirements
    pricing_plan = (
        compiled_request.pricing_plan if compiled_request is not None else None
    ) or (
        select_pricing_method_for_product_ir(
            product_ir,
            preferred_method=preferred_method,
            context_description=payoff_description,
        )
        if product_ir is not None and preferred_method is not None
        else select_pricing_method(
            payoff_description,
            instrument_type=instrument_type,
            model=model,
        )
    )
    _record_platform_event(
        compiled_request,
        "quant_selected_method",
        status="ok",
        details={
            "method": pricing_plan.method,
            "required_market_data": sorted(pricing_plan.required_market_data),
            "sensitivity_support": (
                pricing_plan.sensitivity_support.to_dict()
                if pricing_plan.sensitivity_support is not None
                else {}
            ),
        },
    )
    _append_agent_observation(
        build_meta,
        "quant",
        "decision",
        f"Selected pricing method `{pricing_plan.method}`",
        details={
            "required_market_data": sorted(pricing_plan.required_market_data),
            "method_modules": list(pricing_plan.method_modules),
            "reasoning": pricing_plan.reasoning,
            "sensitivity_support": (
                pricing_plan.sensitivity_support.to_dict()
                if pricing_plan.sensitivity_support is not None
                else {}
            ),
        },
    )

    # Step 2: Check market data availability (early — before writing code)
    if market_state is not None:
        data_errors = check_data_availability(pricing_plan, market_state)
        if data_errors:
            _record_platform_event(
                compiled_request,
                "request_failed" if _finalizes_in_executor(compiled_request) else "build_failed",
                status="error",
                success=False if _finalizes_in_executor(compiled_request) else None,
                outcome="request_failed" if _finalizes_in_executor(compiled_request) else None,
                details={
                    "reason": "missing_market_data",
                    "data_errors": data_errors,
                },
            )
            raise MissingCapabilityError(
                pricing_plan.required_market_data - market_state.available_capabilities,
                market_state.available_capabilities,
                details=data_errors,
            )

    # Use quant agent's data requirements if caller didn't specify
    if requirements is None:
        requirements = pricing_plan.required_market_data

    # Step 3: Plan (spec schema + module path)
    plan = plan_build(
        payoff_description,
        requirements,
        model=model,
        instrument_type=instrument_type or getattr(product_ir, "instrument", None),
        preferred_method=pricing_plan.method,
    )
    _record_platform_event(
        compiled_request,
        "planner_completed",
        status="ok",
        details={
            "module_path": plan.steps[0].module_path if plan.steps else "",
            "spec_name": plan.spec_schema.spec_name if plan.spec_schema is not None else "",
            "class_name": plan.spec_schema.class_name if plan.spec_schema is not None else "",
        },
    )

    # Step 3b: Reuse supported deterministic adapters and explicit cached builds.
    existing = _try_import_existing(plan)
    reuse_reason = None
    if existing is not None:
        if _is_deterministic_supported_route(plan):
            reuse_reason = "deterministic_supported_route"
        elif not force_rebuild:
            reuse_reason = "cached_generated_module"
    if existing is not None and reuse_reason is not None:
        _record_platform_event(
            compiled_request,
            "existing_generated_module_reused",
            status="ok",
            details={
                "module_path": plan.steps[0].module_path if plan.steps else "",
                "class_name": getattr(existing, "__name__", type(existing).__name__),
                "reason": reuse_reason,
            },
        )
        return existing

    # Step 4: Design spec
    if plan.spec_schema is not None:
        spec_schema = plan.spec_schema
        _record_platform_event(
            compiled_request,
            "spec_design_skipped",
            status="info",
            details={
                "reason": "deterministic_spec_schema",
                "spec_name": spec_schema.spec_name,
                "class_name": spec_schema.class_name,
                "field_count": len(spec_schema.fields),
            },
        )
    else:
        stage_model = get_model_for_stage("spec_design", model)
        _record_platform_event(
            compiled_request,
            "spec_design_started",
            status="info",
            details={
                "model": stage_model,
                "requirement_count": len(requirements),
                "description_chars": len(payoff_description),
            },
        )
        try:
            with llm_usage_stage("spec_design", metadata={"model": stage_model}) as usage_records:
                spec_schema = _design_spec(payoff_description, requirements, stage_model)
            _record_platform_event(
                compiled_request,
                "spec_design_completed",
                status="ok",
                details={
                    "model": stage_model,
                    "spec_name": spec_schema.spec_name,
                    "class_name": spec_schema.class_name,
                    "field_count": len(spec_schema.fields),
                    "token_usage": summarize_llm_usage(usage_records),
                },
            )
        except Exception as exc:
            _record_platform_event(
                compiled_request,
                "spec_design_failed",
                status="error",
                details={
                    "model": stage_model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        enforce_llm_token_budget(stage="spec_design")

    # Step 5: Generate skeleton
    skeleton = _generate_skeleton(spec_schema, payoff_description)

    # Step 6-9: Generate code with method guidance, validate, retry
    reference_modules = _reference_modules(pricing_plan)
    reference_sources = _gather_references(reference_modules)
    generation_plan = (
        compiled_request.generation_plan if compiled_request is not None else None
    ) or build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=tuple(module_path for module_path, _ in reference_modules),
        product_ir=product_ir,
    )
    if generation_plan.primitive_plan is not None and generation_plan.primitive_plan.blockers:
        blocker_details = {
            "blockers": list(generation_plan.primitive_plan.blockers),
        }
        blocker_chunks = []
        if generation_plan.blocker_report is not None:
            blocker_details["blocker_codes"] = [
                blocker.id for blocker in generation_plan.blocker_report.blockers
            ]
            blocker_details["blocker_report"] = {
                "summary": generation_plan.blocker_report.summary,
                "should_block": generation_plan.blocker_report.should_block,
                "blockers": [asdict(blocker) for blocker in generation_plan.blocker_report.blockers],
            }
            blocker_chunks.append(render_blocker_report(generation_plan.blocker_report))
        else:
            blocker_chunks.append(", ".join(generation_plan.primitive_plan.blockers))
        if generation_plan.new_primitive_workflow is not None:
            from trellis.agent.new_primitive_workflow import render_new_primitive_workflow

            blocker_details["new_primitive_workflow"] = {
                "summary": generation_plan.new_primitive_workflow.summary,
                "items": [asdict(item) for item in generation_plan.new_primitive_workflow.items],
            }
            blocker_chunks.append(
                render_new_primitive_workflow(generation_plan.new_primitive_workflow)
            )
        if build_meta is not None:
            build_meta["blocker_details"] = blocker_details
        _record_platform_event(
            compiled_request,
            "request_blocked" if _finalizes_in_executor(compiled_request) else "build_blocked",
            status="error",
            success=False if _finalizes_in_executor(compiled_request) else None,
            outcome="request_blocked" if _finalizes_in_executor(compiled_request) else None,
            details=blocker_details,
        )
        raise RuntimeError(
            "Cannot generate payoff because primitive planning blockers remain:\n"
            + "\n\n".join(blocker_chunks)
        )

    ensure_agent_package()
    step = plan.steps[0]
    module_name = f"trellis.{step.module_path.replace('/', '.').replace('.py', '')}"
    _record_platform_event(
        compiled_request,
        "build_started",
        status="info",
        details={
            "module_name": module_name,
            "validation": validation,
            "max_retries": max_retries,
        },
    )
    if validation != "thorough":
        _record_platform_event(
            compiled_request,
            "model_validator_skipped",
            status="info",
            details={"validation": validation},
        )

    # Retrieve all relevant knowledge for this task (single call)
    validation_feedback = ""
    payoff_cls = None
    previous_failures = []
    retry_reason = None
    for attempt in range(max_retries):
        attempt_number = attempt + 1
        prompt_surface = _builder_prompt_surface_for_attempt(
            attempt_number=attempt_number,
            retry_reason=retry_reason,
        )
        knowledge_text, knowledge_surface = _builder_knowledge_context_for_attempt(
            pricing_plan,
            instrument_type,
            attempt_number=attempt_number,
            retry_reason=retry_reason,
            compiled_request=compiled_request,
            product_ir=product_ir,
        )
        _record_platform_event(
            compiled_request,
            "builder_attempt_started",
            status="info",
            details={
                "attempt": attempt_number,
                "prompt_surface": prompt_surface,
                "knowledge_surface": knowledge_surface,
                "retry_reason": retry_reason,
                "knowledge_context_chars": len(knowledge_text),
            },
        )
        stage_model = get_model_for_stage("code_generation", model)
        try:
            with llm_usage_stage(
                "code_generation",
                metadata={"attempt": attempt_number, "model": stage_model},
            ) as usage_records:
                code = _generate_module(
                    skeleton, spec_schema, reference_sources, stage_model, 1,
                    extra_context=validation_feedback,
                    pricing_plan=pricing_plan,
                    knowledge_context=knowledge_text,
                    generation_plan=generation_plan,
                    prompt_surface=prompt_surface,
                )
        except Exception as exc:
            generation_token_usage = summarize_llm_usage(locals().get("usage_records"))
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "code_generation",
                    "model": stage_model,
                    "failure_count": 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "token_usage": generation_token_usage,
                },
            )
            raise
        generation_token_usage = summarize_llm_usage(usage_records)
        enforce_llm_token_budget(stage="code_generation")
        _record_platform_event(
            compiled_request,
            "builder_attempt_generated",
            status="ok",
            details={
                "attempt": attempt_number,
                "prompt_surface": prompt_surface,
                "knowledge_surface": knowledge_surface,
                "retry_reason": retry_reason,
                "knowledge_context_chars": len(knowledge_text),
                "token_usage": generation_token_usage,
            },
        )

        import_report = validate_generated_imports(code, generation_plan)
        if not import_report.ok:
            failures = list(import_report.errors)
            previous_failures = failures
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "import_validation",
                    "failure_count": len(failures),
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## IMPORT VALIDATION FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nUse only approved, registry-backed Trellis imports."
            )
            retry_reason = "import_validation"
            continue

        semantic_report = validate_semantics(
            code,
            product_ir=product_ir,
            generation_plan=generation_plan,
        )
        if not semantic_report.ok:
            failures = list(semantic_report.errors)
            previous_failures = failures
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "semantic_validation",
                    "failure_count": len(failures),
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## SEMANTIC VALIDATION FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nMatch the generated code to the product semantics and real Trellis engine contracts."
            )
            retry_reason = "semantic_validation"
            continue

        from trellis.agent.lite_review import review_generated_code

        lite_review_report = review_generated_code(
            code,
            pricing_plan=pricing_plan,
            product_ir=product_ir,
            generation_plan=generation_plan,
        )
        _record_platform_event(
            compiled_request,
            "lite_review_completed",
            status="ok" if lite_review_report.ok else "error",
            details={
                "attempt": attempt_number,
                "issue_count": len(lite_review_report.issues),
                "issue_codes": [issue.code for issue in lite_review_report.issues],
            },
        )
        if not lite_review_report.ok:
            failures = list(lite_review_report.errors)
            previous_failures = failures
            for issue in lite_review_report.issues:
                _append_agent_observation(
                    build_meta,
                    "lite_reviewer",
                    "failure",
                    issue.message,
                    severity="error",
                    attempt=attempt_number,
                    details={"code": issue.code},
                )
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "lite_review",
                    "failure_count": len(failures),
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## DETERMINISTIC LITE REVIEW FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nRemove hardcoded market inputs and read required pricing inputs from `market_state`."
            )
            retry_reason = "lite_review"
            continue

        file_path = write_module(step.module_path, code)
        mod = dynamic_import(file_path, module_name)
        payoff_cls = getattr(mod, spec_schema.class_name)

        if validation == "fast":
            _record_platform_event(
                compiled_request,
                "builder_attempt_succeeded",
                status="ok",
                details={"attempt": attempt_number},
            )
            _record_platform_event(
                compiled_request,
                "build_completed",
                status="ok",
                success=True if _finalizes_in_executor(compiled_request) else None,
                outcome="build_completed" if _finalizes_in_executor(compiled_request) else None,
                details={"attempts": attempt_number},
            )
            return payoff_cls

        failures = _validate_build(
            payoff_cls, code, payoff_description, spec_schema,
            validation=validation,
            model=model,
            compiled_request=compiled_request,
            pricing_plan=pricing_plan,
            product_ir=product_ir,
            build_meta=build_meta,
            attempt_number=attempt_number,
        )

        if not failures:
            _record_platform_event(
                compiled_request,
                "builder_attempt_succeeded",
                status="ok",
                details={"attempt": attempt_number},
            )
            # Success — record lessons from any failures resolved this round
            if previous_failures:
                _record_resolved_failures(
                    previous_failures, payoff_description, pricing_plan, model,
                )
            _record_platform_event(
                compiled_request,
                "build_completed",
                status="ok",
                success=True if _finalizes_in_executor(compiled_request) else None,
                outcome="build_completed" if _finalizes_in_executor(compiled_request) else None,
                details={"attempts": attempt_number},
            )
            return payoff_cls

        # Diagnose failures and enrich feedback for next attempt
        diagnosis_text = _diagnose_and_enrich(failures)
        previous_failures = failures
        _record_platform_event(
            compiled_request,
            "builder_attempt_failed",
            status="error",
            details={
                "attempt": attempt_number,
                "reason": "validation",
                "failure_count": len(failures),
            },
        )

        # Record run trace (cold storage)
        _record_trace(instrument_type, pricing_plan, payoff_description,
                      attempt, code, failures)

        validation_feedback = (
            "\n\n## VALIDATION FAILURES (your previous code had these issues):\n"
            + "\n".join(f"- {f}" for f in failures)
            + diagnosis_text
            + "\n\nFix ALL of the above issues in your implementation."
        )
        retry_reason = "validation"

    if payoff_cls is not None:
        return payoff_cls

    failure_text = "; ".join(previous_failures) if previous_failures else "unknown build failure"
    _record_platform_event(
        compiled_request,
        "request_failed" if _finalizes_in_executor(compiled_request) else "build_failed",
        status="error",
        success=False if _finalizes_in_executor(compiled_request) else None,
        outcome="request_failed" if _finalizes_in_executor(compiled_request) else None,
        details={
            "reason": "max_retries_exhausted",
            "failure_count": len(previous_failures),
        },
    )
    raise RuntimeError(
        f"Failed to build payoff after {max_retries} attempts: {failure_text}"
    )


def _validate_build(
    payoff_cls,
    code: str,
    description: str,
    spec_schema,
    validation: str = "standard",
    model: str | None = None,
    compiled_request=None,
    pricing_plan=None,
    product_ir=None,
    build_meta: dict | None = None,
    attempt_number: int | None = None,
) -> list[str]:
    """Run validation checks on a built payoff. Returns list of failures."""
    from trellis.agent.review_policy import determine_review_policy
    from trellis.agent.validation_bundles import (
        execute_validation_bundle,
        select_validation_bundle,
    )
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol
    from trellis.core.payoff import DeterministicCashflowPayoff
    from trellis.instruments.bond import Bond

    settle = date(2024, 11, 15)
    failures = []
    itype = _extract_instrument_type(description)
    required_market_data = set(getattr(pricing_plan, "required_market_data", ()) or ())
    review_policy = determine_review_policy(
        validation=validation,
        method=(pricing_plan.method if pricing_plan is not None else "unknown"),
        instrument_type=itype,
        product_ir=product_ir,
    )
    review_knowledge_text = ""
    review_prompt_surface = "none"

    # Try to instantiate the payoff with default test parameters
    try:
        test_payoff = _make_test_payoff(payoff_cls, spec_schema, settle)
    except Exception as e:
        failures.append(f"Cannot instantiate payoff for validation: {e}")
        return failures

    def payoff_factory():
        """Instantiate a fresh payoff under the generated spec schema."""
        return _make_test_payoff(payoff_cls, spec_schema, settle)

    bond = Bond(
        face=100, coupon=0.05,
        maturity_date=date(2034, 11, 15),
        maturity=10, frequency=2,
    )

    def reference_factory():
        """Build the straight-bond reference payoff used for bounding checks."""
        return DeterministicCashflowPayoff(bond)

    def _build_validation_market_state(rate=0.05, vol=0.20):
        """Create a simple market state matching the plan's capability contract."""
        discount_curve = YieldCurve.flat(rate)
        payload = {
            "as_of": settle,
            "settlement": settle,
            "discount": discount_curve,
        }
        if "forward_curve" in required_market_data:
            from trellis.curves.forward_curve import ForwardCurve

            payload["forward_curve"] = ForwardCurve(discount_curve)
        if "black_vol_surface" in required_market_data:
            payload["vol_surface"] = FlatVol(vol)
        if "spot" in required_market_data:
            payload["spot"] = 100.0
            payload["underlier_spots"] = {"SPX": 100.0}
        if "local_vol_surface" in required_market_data:
            def local_vol_surface(spot, time, level=vol):
                from trellis.core.differentiable import get_numpy

                np = get_numpy()
                spot_array = np.asarray(spot, dtype=float)
                time_array = np.asarray(time, dtype=float)
                smile = 1.0 + 0.10 * np.abs(np.log(np.maximum(spot_array, 1e-8) / 100.0))
                term = 1.0 + 0.05 * np.minimum(np.maximum(time_array, 0.0), 5.0) / 5.0
                return level * smile * term

            payload["local_vol_surface"] = local_vol_surface
            payload["local_vol_surfaces"] = {"spx_local_vol": local_vol_surface}
            payload.setdefault("spot", 100.0)
            payload.setdefault("underlier_spots", {"SPX": 100.0})
        return MarketState(**payload)

    ms = _build_validation_market_state()

    def ms_factory(rate=0.05, vol=0.20):
        """Create simple market states for invariant checks."""
        return _build_validation_market_state(rate=rate, vol=vol)

    validation_bundle = select_validation_bundle(
        instrument_type=itype,
        method=(pricing_plan.method if pricing_plan is not None else "unknown"),
        product_ir=product_ir,
    )
    _record_platform_event(
        compiled_request,
        "validation_bundle_selected",
        status="info",
        details={
            "bundle_id": validation_bundle.bundle_id,
            "checks": list(validation_bundle.checks),
            "categories": {
                key: list(value) for key, value in validation_bundle.categories.items()
            },
        },
    )
    bundle_execution = execute_validation_bundle(
        validation_bundle,
        validation_level=validation,
        test_payoff=test_payoff,
        market_state=ms,
        payoff_factory=payoff_factory,
        market_state_factory=ms_factory,
        reference_factory=reference_factory,
    )
    failures.extend(bundle_execution.failures)
    _record_platform_event(
        compiled_request,
        "validation_bundle_executed",
        status="ok" if not bundle_execution.failures else "error",
        details={
            "bundle_id": validation_bundle.bundle_id,
            "executed_checks": list(bundle_execution.executed_checks),
            "skipped_checks": list(bundle_execution.skipped_checks),
            "failure_count": len(bundle_execution.failures),
        },
    )

    deterministic_gate_failed = bool(failures)
    deterministic_gate_reason = (
        "deterministic_validation_failed" if deterministic_gate_failed else None
    )

    # Standard: run critic
    if (
        validation in ("standard", "thorough")
        and review_policy.run_critic
        and not deterministic_gate_failed
    ):
        review_knowledge_text, review_prompt_surface = _review_knowledge_context_for_attempt(
            pricing_plan,
            itype,
            attempt_number=attempt_number or 1,
            compiled_request=compiled_request,
            product_ir=product_ir,
        )
        try:
            from trellis.agent.critic import critique
            from trellis.agent.arbiter import run_critic_tests
            stage_model = get_model_for_stage("critic", model)
            with llm_usage_stage(
                "critic",
                metadata={"attempt": attempt_number, "model": stage_model},
            ) as usage_records:
                concerns = critique(
                    code,
                    description,
                    knowledge_context=review_knowledge_text,
                    model=stage_model,
                )
            enforce_llm_token_budget(stage="critic")
            _record_platform_event(
                compiled_request,
                "critic_completed",
                status="ok",
                details={
                    "concern_count": len(concerns),
                    "prompt_surface": review_prompt_surface,
                    "knowledge_context_chars": len(review_knowledge_text),
                    "token_usage": summarize_llm_usage(usage_records),
                },
            )
            for concern in concerns:
                _append_agent_observation(
                    build_meta,
                    "critic",
                    "concern",
                    concern.description,
                    severity=concern.severity,
                    attempt=attempt_number,
                    details={"test_code": concern.test_code},
                )
            critic_failures = run_critic_tests(concerns, test_payoff)
            failures.extend(critic_failures)
            for failure in critic_failures:
                _append_agent_observation(
                    build_meta,
                    "arbiter",
                    "failure",
                    failure.splitlines()[0],
                    severity="error",
                    attempt=attempt_number,
                    details={"message": failure},
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Critic validation error (non-blocking): {e}")
    elif validation in ("standard", "thorough"):
        _record_platform_event(
            compiled_request,
            "critic_skipped",
            status="info",
            details={
                "risk_level": review_policy.risk_level,
                "reason": deterministic_gate_reason or review_policy.critic_reason,
                "failure_count": len(failures),
            },
        )

    _record_platform_event(
        compiled_request,
        "arbiter_completed",
        status="ok" if not failures else "error",
        details={
            "validation": validation,
            "failure_count": len(failures),
        },
    )

    # Thorough: run model validator (MRM-style)
    if validation == "thorough" and not deterministic_gate_failed:
        try:
            from trellis.agent.model_validator import validate_model

            if review_policy.run_model_validator_llm and not review_knowledge_text:
                review_knowledge_text, review_prompt_surface = _review_knowledge_context_for_attempt(
                    pricing_plan,
                    itype,
                    attempt_number=attempt_number or 1,
                    compiled_request=compiled_request,
                    product_ir=product_ir,
                )

            usage_summary = None
            if review_policy.run_model_validator_llm:
                stage_model = get_model_for_stage("model_validator", model)
                with llm_usage_stage(
                    "model_validator",
                    metadata={"attempt": attempt_number, "model": stage_model},
                ) as usage_records:
                    report = validate_model(
                        payoff_factory=payoff_factory,
                        market_state_factory=ms_factory,
                        code=code,
                        instrument_type=itype,
                        method=spec_schema.requirements[0] if hasattr(spec_schema, 'requirements') else "unknown",
                        knowledge_context=review_knowledge_text,
                        model=stage_model,
                        product_ir=product_ir,
                        run_llm_review=True,
                    )
                enforce_llm_token_budget(stage="model_validator")
                usage_summary = summarize_llm_usage(usage_records)
            else:
                _record_platform_event(
                    compiled_request,
                    "model_validator_llm_review_skipped",
                    status="info",
                    details={
                        "risk_level": review_policy.risk_level,
                        "reason": review_policy.model_validator_reason,
                    },
                )
                report = validate_model(
                    payoff_factory=payoff_factory,
                    market_state_factory=ms_factory,
                    code=code,
                    instrument_type=itype,
                    method=spec_schema.requirements[0] if hasattr(spec_schema, 'requirements') else "unknown",
                    knowledge_context="",
                    model=model,
                    product_ir=product_ir,
                    run_llm_review=False,
                )

            for finding in report.findings:
                _append_agent_observation(
                    build_meta,
                    "model_validator",
                    "finding",
                    finding.description,
                    severity=finding.severity,
                    attempt=attempt_number,
                    details={
                        "category": finding.category,
                        "evidence": finding.evidence,
                        "remediation": finding.remediation,
                    },
                )
                if finding.severity in ("critical", "high"):
                    failures.append(
                        f"[MODEL VALIDATION {finding.severity.upper()}] {finding.id}: "
                        f"{finding.description}\n"
                        f"  Evidence: {finding.evidence}\n"
                        f"  Remediation: {finding.remediation}"
                    )
            blocker_findings = [
                finding for finding in report.findings
                if finding.severity in ("critical", "high")
            ]
            _record_platform_event(
                compiled_request,
                "model_validator_completed",
                status="ok" if not blocker_findings else "error",
                details={
                    "finding_count": len(report.findings),
                    "blocker_count": len(blocker_findings),
                    "approved": report.approved,
                    "llm_review": review_policy.run_model_validator_llm,
                    "risk_level": review_policy.risk_level,
                    "skip_reason": (
                        None if review_policy.run_model_validator_llm
                        else review_policy.model_validator_reason
                    ),
                    "prompt_surface": review_prompt_surface if review_policy.run_model_validator_llm else "deterministic_only",
                    "knowledge_context_chars": len(review_knowledge_text) if review_policy.run_model_validator_llm else 0,
                    "token_usage": usage_summary,
                },
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Model validation error (non-blocking): {e}")
    elif validation == "thorough":
        _record_platform_event(
            compiled_request,
            "model_validator_llm_review_skipped",
            status="info",
            details={
                "risk_level": review_policy.risk_level,
                "reason": deterministic_gate_reason,
                "failure_count": len(failures),
            },
        )

    return failures


def _extract_instrument_type(description: str) -> str:
    """Extract instrument type keyword from a description."""
    desc = description.lower()
    for keyword in ["callable_bond", "callable bond", "puttable_bond", "bermudan_swaption",
                     "barrier_option", "barrier option", "asian_option", "cdo",
                     "nth_to_default", "swaption", "cap", "floor", "swap", "bond", "mbs"]:
        if keyword.replace("_", " ") in desc or keyword in desc:
            return keyword.replace(" ", "_")
    return "unknown"


def _make_test_payoff(payoff_cls, spec_schema, settle: date):
    """Create a test payoff instance from the spec schema with default values."""
    import sys

    spec_cls = None
    module = sys.modules.get(getattr(payoff_cls, "__module__", ""))
    if module is not None and hasattr(module, spec_schema.spec_name):
        spec_cls = getattr(module, spec_schema.spec_name)
    else:
        for _, mod in list(sys.modules.items()):
            if mod and hasattr(mod, spec_schema.spec_name):
                spec_cls = getattr(mod, spec_schema.spec_name)
                break
    if spec_cls is None:
        raise RuntimeError(f"Cannot find {spec_schema.spec_name} in loaded modules")

    # Build kwargs from field definitions with test defaults
    kwargs = {}
    type_defaults = {
        "float": 100.0,
        "int": 10,
        "str": "test",
        "bool": True,
        "date": date(2034, 11, 15),
        "str | None": None,
        "Frequency": None,  # use dataclass default
        "DayCountConvention": None,  # use dataclass default
    }
    # More specific field-name defaults
    name_defaults = {
        "notional": 100.0,
        "coupon": 0.05,
        "strike": 0.05,
        "expiry_date": date(2025, 11, 15),
        "swap_start": date(2025, 11, 15),
        "swap_end": date(2034, 11, 15),
        "start_date": settle,
        "end_date": date(2034, 11, 15),
        "is_payer": True,
        "call_dates": "2027-11-15,2029-11-15,2031-11-15",
        "put_dates": "2027-11-15,2029-11-15,2031-11-15",
        "exercise_dates": "2027-11-15,2029-11-15,2031-11-15",
        "call_price": 100.0,
        "put_price": 100.0,
        "call_schedule": "2027-11-15,2029-11-15,2031-11-15",
        "barrier": 80.0,
        "barrier_type": "down_and_out",
        "option_type": "call",
        "spot": 100.0,
        "fx_pair": "EURUSD",
        "foreign_discount_key": "EUR-DISC",
        "n_names": 5,
        "n_th": 1,
        "attachment": 0.03,
        "detachment": 0.07,
        "correlation": 0.3,
        "recovery": 0.4,
    }

    for field in spec_schema.fields:
        if field.name in name_defaults:
            kwargs[field.name] = name_defaults[field.name]
        elif field.default is not None:
            pass  # let the dataclass default handle it
        elif field.type in type_defaults:
            kwargs[field.name] = type_defaults[field.type]

    try:
        spec = spec_cls(**kwargs)
    except TypeError:
        # If we're missing required fields, try with all defaults
        spec = spec_cls(**{f.name: name_defaults.get(f.name, type_defaults.get(f.type, ""))
                          for f in spec_schema.fields if f.default is None})

    return payoff_cls(spec)


def _design_spec(
    payoff_description: str,
    requirements: set[str],
    model: str,
):
    """LLM call #1: design the spec schema via structured JSON output."""
    from trellis.agent.config import llm_generate_json, ALLOWED_FIELD_TYPES
    from trellis.agent.prompts import spec_design_prompt
    from trellis.agent.planner import SpecSchema, FieldDef

    prompt = spec_design_prompt(payoff_description, requirements)
    data = llm_generate_json(prompt, model=model)

    fields = []
    for f in data["fields"]:
        ftype = f["type"]
        if ftype not in ALLOWED_FIELD_TYPES:
            ftype = "str"  # fallback
        fields.append(FieldDef(
            name=f["name"],
            type=ftype,
            description=f.get("description", ""),
            default=f.get("default"),
        ))

    return SpecSchema(
        class_name=data["class_name"],
        spec_name=data["spec_name"],
        requirements=data["requirements"],
        fields=fields,
    )


def _generate_skeleton(spec_schema, description: str) -> str:
    """Deterministically generate the full module skeleton from the spec schema."""
    required = [f for f in spec_schema.fields if f.default is None]
    optional = [f for f in spec_schema.fields if f.default is not None]
    field_lines = []
    for f in required + optional:
        if f.default is None:
            field_lines.append(f"    {f.name}: {f.type}")
        else:
            field_lines.append(f"    {f.name}: {f.type} = {f.default}")
    fields_block = "\n".join(field_lines)

    requirements_str = ", ".join(f'"{r}"' for r in sorted(spec_schema.requirements))

    return f'''"""Agent-generated payoff: {description}."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class {spec_schema.spec_name}:
    """Specification for {description}."""
{fields_block}


class {spec_schema.class_name}:
    """{description}."""

    def __init__(self, spec: {spec_schema.spec_name}):
        self._spec = spec

    @property
    def spec(self) -> {spec_schema.spec_name}:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {{{requirements_str}}}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
{EVALUATE_SENTINEL}
'''


def _generate_module(
    skeleton: str,
    spec_schema,
    reference_sources: dict[str, str],
    model: str,
    max_retries: int,
    extra_context: str = "",
    pricing_plan=None,
    knowledge_context: str = "",
    generation_plan=None,
    prompt_surface: str = "expanded",
) -> str:
    """LLM call #2: generate the complete module with evaluate() filled in."""
    from trellis.agent.config import llm_generate
    from trellis.agent.prompts import evaluate_prompt

    prompt = evaluate_prompt(skeleton, spec_schema, reference_sources,
                             pricing_plan=pricing_plan,
                             knowledge_context=knowledge_context,
                             generation_plan=generation_plan,
                             prompt_surface=prompt_surface)
    if extra_context:
        prompt += extra_context

    last_error = ""
    for attempt in range(max_retries):
        if attempt > 0:
            full_prompt = prompt + f"\n\n## Previous attempt had error:\n{last_error}\nFix the code."
        else:
            full_prompt = prompt

        try:
            code = llm_generate(full_prompt, model=model)

            # Strip markdown fences
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            code = code.expandtabs(4)
            if not code.strip():
                raise RuntimeError("LLM returned empty module body")
            compile(code, "<agent>", "exec")
            return code
        except SyntaxError as e:
            last_error = str(e)
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Agent failed to produce valid module after {max_retries} attempts"
                ) from e
        except RuntimeError as e:
            last_error = str(e)
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Agent failed to produce valid module after {max_retries} attempts: {last_error}"
                ) from e

    raise RuntimeError("Unreachable")


def _normalize_indent(code: str, target: int = 8) -> str:
    """Re-indent code so the base level is *target* spaces, preserving relative indent.

    Uses textwrap.dedent to strip common leading whitespace first,
    then prepends *target* spaces to each line.
    """
    import textwrap
    dedented = textwrap.dedent(code)
    lines = dedented.split("\n")
    new_lines = []
    for line in lines:
        if line.strip():
            new_lines.append(" " * target + line)
        else:
            new_lines.append("")
    return "\n".join(new_lines)


def _combine_skeleton_and_body(skeleton: str, evaluate_body: str) -> str:
    """Replace the evaluate sentinel in the skeleton with the generated body."""
    if EVALUATE_SENTINEL not in skeleton:
        raise ValueError("Skeleton does not contain evaluate sentinel")
    return skeleton.replace(EVALUATE_SENTINEL, evaluate_body)


def _try_import_existing(plan) -> type | None:
    """Try to import a previously built payoff class."""
    from trellis.agent.builder import TRELLIS_ROOT, dynamic_import

    for step in plan.steps:
        file_path = TRELLIS_ROOT / step.module_path
        if not file_path.exists():
            return None

    last_step = plan.steps[-1]
    file_path = TRELLIS_ROOT / last_step.module_path
    module_name = f"trellis.{last_step.module_path.replace('/', '.').replace('.py', '')}"

    try:
        mod = dynamic_import(file_path, module_name)
        return getattr(mod, plan.payoff_class_name, None)
    except Exception:
        return None


def _is_deterministic_supported_route(plan) -> bool:
    """Whether the plan targets a checked-in deterministic adapter."""
    return bool(plan.steps) and all(
        step.module_path in _DETERMINISTIC_SUPPORTED_ROUTE_MODULES
        for step in plan.steps
    )


def _reference_modules(pricing_plan=None) -> tuple[tuple[str, str], ...]:
    """Select authoritative reference modules for prompt grounding."""
    from trellis.agent.knowledge.methods import normalize_method

    modules = [
        ("trellis.core.payoff", "Payoff protocol + Cashflows/PresentValue return types"),
        ("trellis.core.date_utils", "Date utilities used by generated payoffs"),
        ("trellis.core.market_state", "MarketState capabilities and access patterns"),
        ("trellis.core.types", "Frequency/day-count types"),
        ("trellis.models.black", "Black-style analytical helpers"),
    ]

    if pricing_plan:
        method = normalize_method(pricing_plan.method)
        if method == "analytical":
            modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))
        elif method == "rate_tree":
            modules.append(("trellis.instruments.callable_bond", "CallableBondPayoff (tree reference)"))
            modules.append(("trellis.models.trees", "Tree package exports"))
        elif method == "monte_carlo":
            modules.append(("trellis.instruments.barrier_option", "BarrierOptionPayoff (MC reference)"))
            modules.append(("trellis.models.monte_carlo", "Monte Carlo package exports"))
        elif method == "qmc":
            modules.append(("trellis.models.qmc", "QMC package exports"))
            modules.append(("trellis.models.monte_carlo", "Monte Carlo package exports"))
            modules.append(("trellis.models.processes.gbm", "GBM process reference"))
        elif method == "copula":
            modules.append(("trellis.instruments.nth_to_default", "NthToDefaultPayoff (copula reference)"))
            modules.append(("trellis.models.copulas", "Copula package exports"))
        elif method == "pde_solver":
            modules.append(("trellis.models.pde", "PDE package exports"))
        elif method == "fft_pricing":
            modules.append(("trellis.models.transforms", "Transform package exports"))
            modules.append(("trellis.models.processes.heston", "Heston process reference"))
        elif method == "waterfall":
            modules.append(("trellis.models.cashflow_engine.waterfall", "Cashflow waterfall reference"))
        else:
            modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))
    else:
        modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for module_path, label in modules:
        if module_path in seen:
            continue
        seen.add(module_path)
        deduped.append((module_path, label))
    return tuple(deduped)


def _gather_references(reference_modules: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """Read reference implementations for the code generation prompt.

    Includes method-specific references based on the quant agent's plan.
    """
    refs = {}
    for mod_path, label in reference_modules:
        try:
            refs[label] = read_module_source(mod_path)
        except Exception:
            pass
    return refs


# ---------------------------------------------------------------------------
# Test resolution workflow — diagnose failures and record lessons
# ---------------------------------------------------------------------------

def _diagnose_and_enrich(failures: list[str]) -> str:
    """Run heuristic diagnosis on validation failures and return enriched feedback.

    This gives the builder agent concrete guidance beyond just "fix it",
    drawing on accumulated experience from past debugging sessions.
    """
    from trellis.agent.test_resolution import TestFailure, diagnose_failure

    enrichments = []
    for failure_msg in failures:
        tf = TestFailure(
            test_name="build_validation",
            test_file="executor.py",
            error_type="ValidationFailure",
            error_message=failure_msg,
            expected=None,
            actual=None,
            traceback="",
        )
        diagnosis = diagnose_failure(tf)

        if diagnosis.category != "unknown":
            enrichments.append(
                f"\n**Diagnosis ({diagnosis.category}, {diagnosis.magnitude}):** "
                f"{diagnosis.root_cause}"
            )
            if diagnosis.related_experience:
                enrichments.append(
                    f"  Related past lessons: {', '.join(diagnosis.related_experience)}"
                )

    # Also match against YAML-driven failure signatures
    try:
        from trellis.agent.knowledge.signatures import diagnose_from_signatures
        from trellis.agent.knowledge import get_store
        sig_text = diagnose_from_signatures(
            failures, get_store()._failure_signatures
        )
        if sig_text:
            enrichments.append(sig_text)
    except Exception:
        pass

    if enrichments:
        return "\n\n## AUTOMATED DIAGNOSIS:\n" + "\n".join(enrichments)
    return ""


def _record_resolved_failures(
    failures: list[str],
    description: str,
    pricing_plan,
    model: str,
) -> None:
    """After a successful retry, ask the LLM to distill the lesson learned.

    This records the experience into experience.py so the builder agent
    never repeats the same mistake in future builds.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from trellis.agent.config import llm_generate_json
        from trellis.agent.test_resolution import Lesson, record_lesson

        prompt = f"""You fixed these validation failures for a {description}
({pricing_plan.method if pricing_plan else 'unknown'} method):

{chr(10).join(f'- {f}' for f in failures)}

Distill ONE concise lesson learned. Return JSON:
{{
  "category": "calibration|volatility|backward_induction|finite_differences|monte_carlo|market_data",
  "title": "Short title (max 10 words)",
  "mistake": "What went wrong (1 sentence)",
  "why": "Why it happens — the mental model failure (1-2 sentences)",
  "detect": "How to detect this issue (1 sentence)",
  "fix": "How to fix it (1 sentence)"
}}

Only return JSON, no markdown."""

        data = llm_generate_json(prompt, model=model)
        required = {"category", "title", "mistake", "why", "detect", "fix"}
        missing = required - set(data.keys())
        if missing:
            logger.warning(f"LLM lesson output missing fields: {missing}")
            return

        lesson = Lesson(**data)
        record_lesson(lesson)

        # Also capture into the knowledge system with feature tags
        try:
            from trellis.agent.knowledge.promotion import capture_lesson as kn_capture
            from trellis.agent.knowledge.decompose import decompose as kn_decompose
            features = []
            try:
                decomp = kn_decompose(description, instrument_type=None)
                features = list(decomp.features)
            except Exception:
                pass
            kn_capture(
                category=data.get("category", "unknown"),
                title=data.get("title", ""),
                severity="high",
                symptom=data.get("mistake", data.get("detect", "")),
                root_cause=data.get("why", ""),
                fix=data.get("fix", ""),
                validation=f"Resolved during build of {description}",
                method=pricing_plan.method if pricing_plan else None,
                features=features,
                confidence=0.5,
            )
        except Exception as e:
            logger.warning(f"Knowledge capture failed (non-blocking): {e}")
    except Exception as e:
        logger.warning(f"Lesson recording failed (non-blocking): {e}")


# ---------------------------------------------------------------------------
# Knowledge system helpers
# ---------------------------------------------------------------------------

def _builder_knowledge_context_for_attempt(
    pricing_plan,
    instrument_type: str | None,
    *,
    attempt_number: int,
    retry_reason: str | None = None,
    compiled_request=None,
    product_ir=None,
) -> tuple[str, str]:
    """Return compact knowledge on first pass and expanded context on retries."""
    expanded = attempt_number > 1 and retry_reason == "validation"
    knowledge_surface = "expanded" if expanded else "compact"

    if compiled_request is not None and getattr(compiled_request, "knowledge", None) is not None:
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload

        if not expanded and compiled_request.knowledge_text:
            return compiled_request.knowledge_text, knowledge_surface

        payload = build_shared_knowledge_payload(compiled_request.knowledge)
        key = "builder_text_expanded" if expanded else "builder_text_distilled"
        return payload.get(key, ""), knowledge_surface

    return _retrieve_knowledge(
        pricing_plan,
        instrument_type,
        product_ir=product_ir,
        compact=not expanded,
    ), knowledge_surface


def _review_knowledge_context_for_attempt(
    pricing_plan,
    instrument_type: str | None,
    *,
    attempt_number: int,
    compiled_request=None,
    product_ir=None,
) -> tuple[str, str]:
    """Return reviewer knowledge with the same compact-first policy."""
    return _knowledge_context_for_attempt(
        audience="review",
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        attempt_number=attempt_number,
        compiled_request=compiled_request,
        product_ir=product_ir,
    )


def _builder_prompt_surface_for_attempt(
    *,
    attempt_number: int,
    retry_reason: str | None,
) -> str:
    """Select the retry prompt surface from the previous failure type."""
    if attempt_number <= 1:
        return "compact"
    if retry_reason == "import_validation":
        return "import_repair"
    if retry_reason in {"semantic_validation", "lite_review"}:
        return "semantic_repair"
    return "expanded"


def _knowledge_context_for_attempt(
    *,
    audience: str,
    pricing_plan,
    instrument_type: str | None,
    attempt_number: int,
    compiled_request=None,
    product_ir=None,
) -> tuple[str, str]:
    """Select compact knowledge first, then expand after the first failed attempt."""
    expanded = attempt_number > 1
    prompt_surface = "expanded" if expanded else "compact"

    if compiled_request is not None and getattr(compiled_request, "knowledge", None) is not None:
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload

        if not expanded:
            if audience == "builder" and compiled_request.knowledge_text:
                return compiled_request.knowledge_text, prompt_surface
            if audience == "review" and compiled_request.review_knowledge_text:
                return compiled_request.review_knowledge_text, prompt_surface

        payload = build_shared_knowledge_payload(compiled_request.knowledge)
        if audience == "builder":
            key = "builder_text_expanded" if expanded else "builder_text_distilled"
        elif audience == "review":
            key = "review_text_expanded" if expanded else "review_text_distilled"
        else:
            raise ValueError(f"Unsupported knowledge audience: {audience}")
        return payload.get(key, ""), prompt_surface

    if audience == "builder":
        return _retrieve_knowledge(
            pricing_plan,
            instrument_type,
            product_ir=product_ir,
            compact=not expanded,
        ), prompt_surface
    if audience == "review":
        return _retrieve_review_knowledge(
            pricing_plan,
            instrument_type,
            product_ir=product_ir,
            compact=not expanded,
        ), prompt_surface
    raise ValueError(f"Unsupported knowledge audience: {audience}")


def _retrieve_knowledge(
    pricing_plan,
    instrument_type: str | None,
    *,
    product_ir=None,
    compact: bool = True,
) -> str:
    """Retrieve all relevant knowledge for a task as formatted prompt text."""
    try:
        from trellis.agent.knowledge import (
            build_shared_knowledge_payload,
            retrieve_for_product_ir,
            retrieve_for_task,
        )
        from trellis.agent.knowledge.decompose import decompose

        if product_ir is not None:
            knowledge = retrieve_for_product_ir(
                product_ir,
                preferred_method=pricing_plan.method,
            )
            payload = build_shared_knowledge_payload(knowledge)
            return payload["builder_text_distilled" if compact else "builder_text_expanded"]

        features: list[str] = []
        try:
            decomp = decompose(
                instrument_type or "",
                instrument_type=instrument_type,
            )
            features = list(decomp.features)
        except Exception:
            pass

        knowledge = retrieve_for_task(
            method=pricing_plan.method,
            features=features,
            instrument=instrument_type,
        )
        payload = build_shared_knowledge_payload(knowledge)
        return payload["builder_text_distilled" if compact else "builder_text_expanded"]
    except Exception:
        return ""


def _retrieve_review_knowledge(
    pricing_plan,
    instrument_type: str | None,
    *,
    product_ir=None,
    compact: bool = True,
) -> str:
    """Retrieve shared knowledge for reviewer-style agent prompts."""
    try:
        from trellis.agent.knowledge import (
            build_shared_knowledge_payload,
            retrieve_for_product_ir,
            retrieve_for_task,
        )
        from trellis.agent.knowledge.decompose import decompose

        if product_ir is not None:
            knowledge = retrieve_for_product_ir(
                product_ir,
                preferred_method=pricing_plan.method if pricing_plan is not None else None,
            )
            payload = build_shared_knowledge_payload(knowledge)
            return payload["review_text_distilled" if compact else "review_text_expanded"]

        features: list[str] = []
        try:
            decomp = decompose(
                instrument_type or "",
                instrument_type=instrument_type,
            )
            features = list(decomp.features)
        except Exception:
            pass

        knowledge = retrieve_for_task(
            method=pricing_plan.method if pricing_plan is not None else None,
            features=features,
            instrument=instrument_type,
        )
        payload = build_shared_knowledge_payload(knowledge)
        return payload["review_text_distilled" if compact else "review_text_expanded"]
    except Exception:
        return ""


def _record_trace(
    instrument_type: str | None,
    pricing_plan,
    description: str,
    attempt: int,
    code: str,
    failures: list[str],
) -> None:
    """Record a run trace to cold storage (best-effort)."""
    try:
        from trellis.agent.knowledge.promotion import record_trace
        record_trace(
            instrument=instrument_type or "unknown",
            method=pricing_plan.method if pricing_plan else "unknown",
            description=description,
            pricing_plan={"method": pricing_plan.method} if pricing_plan else {},
            attempt=attempt,
            code=code,
            validation_failures=failures,
            resolved=False,
        )
    except Exception:
        pass
