"""Bounded intra-run learning contracts.

This module deliberately separates candidate knowledge from canonical
knowledge.  A candidate can be used as an ephemeral retry overlay inside an
assisted/remediation task run, but it is not a promoted lesson or cookbook.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class RecoveryMode(str, Enum):
    """Runtime policy for automatic recovery after a target failure."""

    STRICT = "strict"
    ASSISTED = "assisted"
    REMEDIATION = "remediation"


_PROVIDER_FAILURE_MARKERS = (
    "openai request failed",
    "anthropic request failed",
    "llm provider",
    "insufficient_quota",
    "rate limit",
    "request failed after",
    "returned invalid json response",
    "returned empty json response",
    "token budget exceeded",
)

_HONEST_BLOCK_MARKERS = (
    "expected honest block",
    "honest_block",
    "repair_packet",
    "unsupported_path",
    "unsupported composite",
    "requires clarification",
)

_API_CONTRACT_MARKERS = (
    "unexpected keyword argument",
    "missing required positional",
    "required_primitive_missing",
    "has no attribute",
    "not defined",
    "signature",
    "callable",
    "helper",
    "primitive",
)

_COOKBOOK_MARKERS = (
    "cookbook",
    "template",
    "route guidance",
    "wiring",
    "adapter",
)

_MARKET_BINDING_MARKERS = (
    "market_state",
    "market binding",
    "discount_curve",
    "vol_surface",
    "black_vol_surface",
    "heston",
    "spot",
)

_UNEXPECTED_KEYWORD_RE = re.compile(
    r"(?P<callable>[A-Za-z_][\w.]*)\(\)\s+got an unexpected keyword "
    r"argument ['\"](?P<keyword>[^'\"]+)['\"]"
)
_TRELLIS_PATH_RE = re.compile(r"\btrellis(?:\.[A-Za-z_]\w*)+\b")


@dataclass(frozen=True)
class KnowledgePatchCandidate:
    """Ephemeral candidate guidance for one bounded retry."""

    candidate_id: str
    target_id: str
    recovery_mode: RecoveryMode
    patch_type: str
    preferred_method: str | None
    instrument_type: str | None
    confidence: float
    summary: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    guidance: tuple[str, ...] = field(default_factory=tuple)
    structured_evidence: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    repair_obligations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    contract_completeness: float = 0.0
    retryable: bool = False
    skip_reasons: tuple[str, ...] = field(default_factory=tuple)
    lesson_ids: tuple[str, ...] = field(default_factory=tuple)
    source_roles: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recovery_mode"] = self.recovery_mode.value
        return payload


def normalize_recovery_mode(value: str | RecoveryMode | None) -> RecoveryMode:
    """Normalize a recovery-mode input into the internal enum."""
    if isinstance(value, RecoveryMode):
        return value
    text = str(value or RecoveryMode.STRICT.value).strip().lower()
    try:
        return RecoveryMode(text)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in RecoveryMode)
        raise ValueError(
            f"Unsupported recovery_mode {value!r}; expected one of: {allowed}"
        ) from exc


def build_knowledge_patch_candidate(
    *,
    target_id: str,
    preferred_method: str | None,
    instrument_type: str | None,
    recovery_mode: str | RecoveryMode,
    payload: Mapping[str, Any],
) -> KnowledgePatchCandidate | None:
    """Build one retry-overlay candidate from failed target evidence."""
    mode = normalize_recovery_mode(recovery_mode)
    if mode == RecoveryMode.STRICT:
        return None
    if bool(payload.get("success")):
        return None

    failures = _string_tuple(payload.get("failures"))
    reflection = _mapping(payload.get("reflection"))
    gaps = _string_tuple(reflection.get("gaps_identified"))
    lesson_ids = _string_tuple(reflection.get("lesson_captured"))
    observations = _observations(payload.get("agent_observations"))
    structured_evidence = _structured_evidence_from_payload(
        payload=payload,
        failures=failures,
    )
    repair_obligations = _repair_obligations_from_evidence(structured_evidence)
    contract_completeness = _contract_completeness(repair_obligations)
    retryable, skip_reasons = _retry_gate(
        repair_obligations=repair_obligations,
        contract_completeness=contract_completeness,
    )
    evidence = _unique_strings(
        [
            *failures[:5],
            *_observation_summaries(observations)[:5],
            *_structured_evidence_lines(structured_evidence)[:8],
        ]
    )
    guidance = _unique_strings(
        [
            *gaps[:8],
            *_lesson_guidance_lines(lesson_ids)[:5],
            *_repair_obligation_lines(repair_obligations)[:8],
        ]
    )
    combined = "\n".join([*evidence, *guidance]).lower()

    if not combined.strip():
        return None
    if _contains_any(combined, _PROVIDER_FAILURE_MARKERS):
        return None
    if _contains_any(combined, _HONEST_BLOCK_MARKERS):
        return None

    patch_type = _classify_patch_type(combined)
    confidence = _candidate_confidence(
        has_gaps=bool(gaps),
        has_lessons=bool(lesson_ids),
        has_observations=bool(observations),
        has_structured_evidence=bool(structured_evidence),
        combined=combined,
    )
    if confidence < 0.35:
        return None

    summary = _candidate_summary(
        target_id=target_id,
        patch_type=patch_type,
        preferred_method=preferred_method,
        first_guidance=guidance[0] if guidance else None,
        first_failure=failures[0] if failures else None,
    )
    seed = {
        "target_id": target_id,
        "preferred_method": preferred_method,
        "instrument_type": instrument_type,
        "patch_type": patch_type,
        "evidence": evidence,
        "guidance": guidance,
        "structured_evidence": structured_evidence,
        "repair_obligations": repair_obligations,
        "contract_completeness": contract_completeness,
        "retryable": retryable,
        "skip_reasons": skip_reasons,
    }
    candidate_id = hashlib.sha256(
        json.dumps(seed, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]

    return KnowledgePatchCandidate(
        candidate_id=candidate_id,
        target_id=target_id,
        recovery_mode=mode,
        patch_type=patch_type,
        preferred_method=preferred_method,
        instrument_type=instrument_type,
        confidence=confidence,
        summary=summary,
        evidence=tuple(evidence),
        guidance=tuple(guidance),
        structured_evidence=tuple(structured_evidence),
        repair_obligations=tuple(repair_obligations),
        contract_completeness=contract_completeness,
        retryable=retryable,
        skip_reasons=tuple(skip_reasons),
        lesson_ids=tuple(lesson_ids),
        source_roles=tuple(_source_roles(observations)),
    )


def render_knowledge_overlay(
    candidates: Sequence[KnowledgePatchCandidate | Mapping[str, Any]],
) -> str:
    """Render candidate knowledge as a prompt overlay."""
    normalized = [_candidate_from_any(candidate) for candidate in candidates]
    normalized = [candidate for candidate in normalized if candidate is not None]
    if not normalized:
        return ""

    lines = [
        "## Intra-Run Candidate Knowledge Overlay",
        "This guidance is ephemeral and not canonical. Use it only to repair "
        "the exact failure from the immediately previous target attempt.",
    ]
    for candidate in normalized:
        lines.extend(
            [
                "",
                f"### Candidate {candidate.candidate_id}",
                f"- Target: `{candidate.target_id}`",
                f"- Patch type: `{candidate.patch_type}`",
                f"- Preferred method: `{candidate.preferred_method or 'unknown'}`",
                f"- Instrument type: `{candidate.instrument_type or 'unknown'}`",
                f"- Confidence: `{candidate.confidence:.2f}`",
                f"- Contract completeness: `{candidate.contract_completeness:.2f}`",
                f"- Retryable: `{candidate.retryable}`",
                f"- Summary: {candidate.summary}",
            ]
        )
        if candidate.skip_reasons:
            lines.append(f"- Skip reasons: {', '.join(candidate.skip_reasons)}")
        if candidate.source_roles:
            lines.append(f"- Source roles: {', '.join(candidate.source_roles)}")
        if candidate.guidance:
            lines.append("- Candidate corrections:")
            lines.extend(f"  - {item}" for item in candidate.guidance[:8])
        if candidate.repair_obligations:
            lines.append("- Structured repair obligations:")
            lines.extend(
                f"  - {_render_record_inline(item)}"
                for item in candidate.repair_obligations[:8]
            )
        if candidate.structured_evidence:
            lines.append("- Structured evidence:")
            lines.extend(
                f"  - {_render_record_inline(item)}"
                for item in candidate.structured_evidence[:8]
            )
        if candidate.evidence:
            lines.append("- Failure evidence:")
            lines.extend(f"  - {item}" for item in candidate.evidence[:8])
        if candidate.lesson_ids:
            lines.append(f"- Related candidate lessons: {', '.join(candidate.lesson_ids)}")
    return "\n".join(lines)


def payloads_for_overlay(
    candidates: Iterable[KnowledgePatchCandidate],
) -> list[dict[str, Any]]:
    """Return serializable candidate payloads."""
    return [candidate.to_payload() for candidate in candidates]


def _candidate_from_any(
    candidate: KnowledgePatchCandidate | Mapping[str, Any],
) -> KnowledgePatchCandidate | None:
    if isinstance(candidate, KnowledgePatchCandidate):
        return candidate
    if not isinstance(candidate, Mapping):
        return None
    try:
        return KnowledgePatchCandidate(
            candidate_id=str(candidate.get("candidate_id") or ""),
            target_id=str(candidate.get("target_id") or ""),
            recovery_mode=normalize_recovery_mode(candidate.get("recovery_mode")),
            patch_type=str(candidate.get("patch_type") or "knowledge_patch"),
            preferred_method=_optional_str(candidate.get("preferred_method")),
            instrument_type=_optional_str(candidate.get("instrument_type")),
            confidence=float(candidate.get("confidence") or 0.0),
            summary=str(candidate.get("summary") or ""),
            evidence=tuple(_string_tuple(candidate.get("evidence"))),
            guidance=tuple(_string_tuple(candidate.get("guidance"))),
            structured_evidence=tuple(
                _record_tuple(candidate.get("structured_evidence"))
            ),
            repair_obligations=tuple(
                _record_tuple(candidate.get("repair_obligations"))
            ),
            contract_completeness=float(candidate.get("contract_completeness") or 0.0),
            retryable=bool(candidate.get("retryable")),
            skip_reasons=tuple(_string_tuple(candidate.get("skip_reasons"))),
            lesson_ids=tuple(_string_tuple(candidate.get("lesson_ids"))),
            source_roles=tuple(_string_tuple(candidate.get("source_roles"))),
        )
    except Exception:
        return None


def _classify_patch_type(combined: str) -> str:
    if _contains_any(combined, _COOKBOOK_MARKERS) or _contains_any(combined, _API_CONTRACT_MARKERS):
        return "cookbook_patch"
    if _contains_any(combined, _MARKET_BINDING_MARKERS):
        return "market_binding_patch"
    if "validation bundle" in combined or "cross-validation" in combined:
        return "validation_patch"
    return "knowledge_patch"


def _candidate_confidence(
    *,
    has_gaps: bool,
    has_lessons: bool,
    has_observations: bool,
    has_structured_evidence: bool,
    combined: str,
) -> float:
    confidence = 0.35
    if has_gaps:
        confidence += 0.12
    if has_lessons:
        confidence += 0.08
    if has_observations:
        confidence += 0.05
    if has_structured_evidence:
        confidence += 0.1
    if _contains_any(combined, _API_CONTRACT_MARKERS):
        confidence += 0.12
    if _contains_any(combined, _COOKBOOK_MARKERS):
        confidence += 0.06
    return round(min(confidence, 0.75), 2)


def _structured_evidence_from_payload(
    *,
    payload: Mapping[str, Any],
    failures: Sequence[str],
) -> list[dict[str, Any]]:
    """Extract deterministic repair evidence from a failed target payload."""
    records: list[dict[str, Any]] = []
    for failure in failures:
        records.extend(_callable_signature_records(failure))
        records.extend(_required_primitive_records(failure))
    comparison = _comparison_contract_record(payload=payload, failures=failures)
    if comparison:
        records.append(comparison)
    return _unique_records(records)


def _callable_signature_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in _UNEXPECTED_KEYWORD_RE.finditer(text):
        callable_name = match.group("callable")
        keyword = match.group("keyword")
        symbol = _symbol_from_callable_name(callable_name)
        if not symbol:
            continue
        resolved = _resolve_symbol(symbol)
        if resolved:
            record = {
                "kind": "callable_signature",
                "callable": callable_name,
                "symbol": symbol,
                "module": resolved["module"],
                "qualified_name": f"{resolved['module']}.{symbol}",
                "signature": resolved["signature"],
                "unexpected_keyword": keyword,
                "available": True,
            }
        else:
            record = {
                "kind": "callable_signature",
                "callable": callable_name,
                "symbol": symbol,
                "signature": "",
                "unexpected_keyword": keyword,
                "available": False,
            }
        records.append(record)
    return records


def _required_primitive_records(text: str) -> list[dict[str, Any]]:
    if "required_primitive_missing" not in text:
        return []
    records = []
    for primitive in _TRELLIS_PATH_RE.findall(text):
        module, symbol = _split_qualified_symbol(primitive)
        resolved = _resolve_qualified_symbol(module=module, symbol=symbol)
        records.append(
            {
                "kind": "required_primitive",
                "primitive": primitive,
                "module": resolved.get("module") or module,
                "symbol": resolved.get("symbol") or symbol,
                "qualified_name": resolved.get("qualified_name") or primitive,
                "signature": resolved.get("signature") or "",
                "available": bool(resolved.get("available")),
            }
        )
    return records


def _comparison_contract_record(
    *,
    payload: Mapping[str, Any],
    failures: Sequence[str],
) -> dict[str, Any] | None:
    comparison = _mapping(payload.get("comparison"))
    combined = "\n".join([*failures, json.dumps(comparison, default=str)]).lower()
    if not comparison and "comparison" not in combined and "cross-validation" not in combined:
        return None

    validation = _mapping(payload.get("validation"))
    runtime_contract = _mapping(payload.get("runtime_contract"))
    generated_artifact = _mapping(payload.get("generated_artifact"))
    method_prices = _method_prices(
        comparison.get("method_prices")
        or comparison.get("prices")
        or payload.get("method_prices")
        or payload.get("method_results")
    )
    record: dict[str, Any] = {
        "kind": "comparison_contract",
        "status": _first_present(comparison, "status", "comparison_status"),
        "method_prices": method_prices,
        "reference_target": _first_present(
            comparison,
            "reference_target",
            "reference_method",
            fallback=payload.get("reference_target"),
        ),
        "tolerance": _first_present(
            comparison,
            "tolerance",
            "relative_tolerance",
            "absolute_tolerance",
            fallback=payload.get("tolerance"),
        ),
        "selected_route": _first_present(
            runtime_contract,
            "selected_route",
            "route",
            fallback=payload.get("selected_route") or payload.get("route"),
        ),
        "binding": _first_present(
            runtime_contract,
            "binding",
            "backend_binding",
            "binding_id",
            fallback=payload.get("binding"),
        ),
        "validation_bundle": _first_present(
            validation,
            "bundle",
            "validation_bundle",
            fallback=payload.get("validation_bundle")
            or runtime_contract.get("validation_bundle"),
        ),
        "payoff_class": payload.get("payoff_class") or generated_artifact.get("class"),
        "payoff_module": payload.get("payoff_module") or generated_artifact.get("module"),
    }
    clean = {key: value for key, value in record.items() if value not in (None, {}, [])}
    return clean if len(clean) > 1 else None


def _repair_obligations_from_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    for record in evidence:
        kind = str(record.get("kind") or "")
        if kind == "callable_signature":
            obligations.append(
                {
                    "kind": "callable_signature",
                    "symbol": record.get("symbol"),
                    "module": record.get("module"),
                    "signature": record.get("signature"),
                    "do_not_pass": record.get("unexpected_keyword"),
                }
            )
        elif kind == "required_primitive":
            obligations.append(
                {
                    "kind": "required_primitive",
                    "primitive": record.get("primitive"),
                    "module": record.get("module"),
                    "symbol": record.get("symbol"),
                    "available": record.get("available"),
                    "signature": record.get("signature"),
                }
            )
        elif kind == "comparison_contract":
            obligations.append(
                {
                    "kind": "comparison_contract",
                    "selected_route": record.get("selected_route"),
                    "binding": record.get("binding"),
                    "validation_bundle": record.get("validation_bundle"),
                    "reference_target": record.get("reference_target"),
                    "tolerance": record.get("tolerance"),
                }
            )
    return [
        {key: value for key, value in obligation.items() if value not in (None, "", {}, [])}
        for obligation in obligations
    ]


def _structured_evidence_lines(records: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        kind = record.get("kind")
        if kind == "callable_signature":
            signature = record.get("signature") or "<unresolved>"
            keyword = record.get("unexpected_keyword")
            lines.append(
                f"Callable signature: `{record.get('qualified_name') or record.get('symbol')}` "
                f"accepts `{signature}`; unexpected keyword `{keyword}` is not in that contract."
            )
        elif kind == "required_primitive":
            available = "available" if record.get("available") else "missing"
            signature = record.get("signature") or "<unresolved>"
            lines.append(
                f"Required primitive: `{record.get('primitive')}` is {available}; "
                f"signature `{signature}`."
            )
        elif kind == "comparison_contract":
            lines.append(
                "Comparison contract: "
                f"route=`{record.get('selected_route', 'unknown')}`, "
                f"binding=`{record.get('binding', 'unknown')}`, "
                f"validation_bundle=`{record.get('validation_bundle', 'unknown')}`, "
                f"reference=`{record.get('reference_target', 'unknown')}`, "
                f"tolerance=`{record.get('tolerance', 'unknown')}`."
            )
    return lines


def _repair_obligation_lines(records: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        kind = record.get("kind")
        if kind == "callable_signature":
            symbol = record.get("symbol") or "callable"
            signature = record.get("signature") or "<unresolved>"
            keyword = record.get("do_not_pass")
            lines.append(
                f"Call `{symbol}` with exact signature `{signature}`; do not pass "
                f"unsupported keyword `{keyword}`."
            )
        elif kind == "required_primitive":
            primitive = record.get("primitive") or record.get("symbol") or "primitive"
            if record.get("available"):
                lines.append(
                    f"Import and use required primitive `{primitive}` with signature "
                    f"`{record.get('signature') or '<unresolved>'}`."
                )
            else:
                lines.append(
                    f"Required primitive `{primitive}` is not importable; report a "
                    "concrete implementation gap instead of inventing a substitute."
                )
        elif kind == "comparison_contract":
            lines.append(
                "Preserve the comparison contract: "
                f"route=`{record.get('selected_route', 'unknown')}`, "
                f"binding=`{record.get('binding', 'unknown')}`, "
                f"validation_bundle=`{record.get('validation_bundle', 'unknown')}`."
            )
    return lines


def _contract_completeness(records: Sequence[Mapping[str, Any]]) -> float:
    """Score whether a candidate has enough concrete contract evidence to retry."""
    score = 0.0
    for record in records:
        kind = record.get("kind")
        if kind == "callable_signature":
            if record.get("signature") and record.get("do_not_pass"):
                score += 0.6
            elif record.get("signature") or record.get("do_not_pass"):
                score += 0.35
        elif kind == "required_primitive":
            if record.get("primitive") and record.get("available") is True:
                score += 0.6
            elif record.get("primitive"):
                score += 0.25
        elif kind == "comparison_contract":
            comparison_score = 0.0
            if record.get("selected_route") or record.get("binding"):
                comparison_score += 0.25
            if record.get("validation_bundle"):
                comparison_score += 0.2
            if record.get("reference_target") or record.get("tolerance") is not None:
                comparison_score += 0.15
            score += comparison_score
    return round(min(score, 1.0), 2)


def _retry_gate(
    *,
    repair_obligations: Sequence[Mapping[str, Any]],
    contract_completeness: float,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether an assisted retry may use this candidate."""
    reasons: list[str] = []
    if not repair_obligations or contract_completeness <= 0.0:
        reasons.append("missing_structured_repair_obligation")
    for obligation in repair_obligations:
        if (
            obligation.get("kind") == "required_primitive"
            and obligation.get("available") is False
        ):
            reasons.append("required_primitive_not_importable")
    if contract_completeness < 0.5:
        reasons.append("contract_completeness_below_retry_threshold")
    return not reasons, tuple(_unique_strings(reasons))


def _symbol_from_callable_name(callable_name: str) -> str:
    parts = [part for part in str(callable_name or "").split(".") if part]
    if not parts:
        return ""
    if parts[-1] == "__init__" and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def _split_qualified_symbol(path: str) -> tuple[str, str]:
    text = str(path or "").strip().strip("`.,:;()[]{}")
    if "." not in text:
        return "", text
    module, _, symbol = text.rpartition(".")
    return module, symbol


def _resolve_symbol(symbol: str) -> dict[str, Any]:
    for module in _symbol_modules(symbol):
        resolved = _resolve_qualified_symbol(
            module=module,
            symbol=symbol,
            allow_registry_fallback=False,
        )
        if resolved.get("available"):
            return resolved
    return {}


def _resolve_qualified_symbol(
    *,
    module: str,
    symbol: str,
    allow_registry_fallback: bool = True,
) -> dict[str, Any]:
    if module and symbol:
        try:
            imported = importlib.import_module(module)
            obj = getattr(imported, symbol)
            return {
                "module": module,
                "symbol": symbol,
                "qualified_name": f"{module}.{symbol}",
                "signature": _signature_for_object(symbol, obj),
                "available": True,
            }
        except Exception:
            pass

    if allow_registry_fallback and symbol:
        for candidate_module in _symbol_modules(symbol):
            if candidate_module == module:
                continue
            resolved = _resolve_qualified_symbol(
                module=candidate_module,
                symbol=symbol,
                allow_registry_fallback=False,
            )
            if resolved.get("available"):
                return resolved

    return {
        "module": module,
        "symbol": symbol,
        "qualified_name": f"{module}.{symbol}" if module and symbol else symbol,
        "signature": "",
        "available": False,
    }


def _symbol_modules(symbol: str) -> tuple[str, ...]:
    try:
        from trellis.agent.knowledge.import_registry import find_symbol_modules

        return find_symbol_modules(symbol)
    except Exception:
        return ()


def _signature_for_object(symbol: str, obj: Any) -> str:
    try:
        return f"{symbol}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        pass
    if inspect.isclass(obj):
        try:
            parameters = list(inspect.signature(obj.__init__).parameters.values())
            if parameters and parameters[0].name == "self":
                parameters = parameters[1:]
            signature = inspect.Signature(parameters=parameters)
            return f"{symbol}{signature}"
        except (TypeError, ValueError):
            pass
    return f"{symbol}(...)"


def _method_prices(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    prices: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            price = _first_present(item, "price", "value", "model_price")
        else:
            price = item
        if price is not None:
            prices[str(key)] = price
    return prices


def _first_present(
    mapping: Mapping[str, Any],
    *keys: str,
    fallback: Any = None,
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return fallback


def _record_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    return ()


def _unique_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        clean = {key: value for key, value in dict(record).items() if value != ""}
        key = json.dumps(clean, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _render_record_inline(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, default=str)


def _candidate_summary(
    *,
    target_id: str,
    patch_type: str,
    preferred_method: str | None,
    first_guidance: str | None,
    first_failure: str | None,
) -> str:
    source = first_guidance or first_failure or "failed target evidence"
    return (
        f"Retry `{target_id}` with a {patch_type} overlay for "
        f"`{preferred_method or 'unknown'}` based on: {source[:180]}"
    )


def _lesson_guidance_lines(lesson_ids: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for lesson_id in lesson_ids:
        text = _load_lesson_guidance(lesson_id)
        if text:
            lines.append(text)
        else:
            lines.append(f"Review candidate lesson `{lesson_id}` from the previous failed attempt.")
    return lines


def _load_lesson_guidance(lesson_id: str) -> str:
    try:
        import yaml
        from trellis.agent.knowledge.reflect import _KNOWLEDGE_DIR

        path = _KNOWLEDGE_DIR / "lessons" / "entries" / f"{lesson_id}.yaml"
        data = yaml.safe_load(path.read_text()) if path.exists() else None
        if not isinstance(data, Mapping):
            return ""
        title = str(data.get("title") or lesson_id).strip()
        fix = str(data.get("fix") or data.get("remediation") or "").strip()
        root_cause = str(data.get("root_cause") or "").strip()
        parts = [part for part in (title, fix, root_cause) if part]
        return " | ".join(parts)
    except Exception:
        return ""


def _source_roles(observations: Sequence[Mapping[str, Any]]) -> list[str]:
    roles = []
    for observation in observations:
        role = str(observation.get("agent") or "").strip()
        if role and role not in roles:
            roles.append(role)
    return roles


def _observation_summaries(
    observations: Sequence[Mapping[str, Any]],
) -> list[str]:
    summaries = []
    for observation in observations:
        role = str(observation.get("agent") or "agent").strip()
        summary = str(observation.get("summary") or "").strip()
        if summary:
            summaries.append(f"{role}: {summary}")
    return summaries


def _observations(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        text = json.dumps(value, sort_keys=True, default=str)
        return (text,) if text and text != "{}" else ()
    if isinstance(value, Iterable):
        values = []
        for item in value:
            values.extend(_string_tuple(item))
        return tuple(_unique_strings(values))
    text = str(value).strip()
    return (text,) if text else ()


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)
