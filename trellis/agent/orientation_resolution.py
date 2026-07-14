"""Bounded, task-relevant content resolution for hosted runtime-agent roles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml

from trellis.agent.role_orientation import (
    OrientationResource,
    RoleOrientation,
    get_role_orientation,
    render_role_orientation_card,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_KINDS = frozenset({"runtime_contract", "runtime_evidence"})
_DOCUMENT_KINDS = frozenset(
    {"official_docs", "official_docs_index", "support_contract"}
)
_STOP_WORDS = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "price",
        "pricing",
        "review",
        "the",
        "to",
        "under",
        "use",
        "validate",
        "with",
    }
)
_RST_HEADING_UNDERLINE = re.compile(r"^([=\-~^\"`:+*#])\1{2,}\s*$")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RoleOrientationQuery:
    """Typed semantic cues used to resolve one hosted-role context packet."""

    instrument_type: str = ""
    description: str = ""
    method: str = ""
    features: tuple[str, ...] = ()
    model_family: str = ""
    route_ids: tuple[str, ...] = ()
    route_families: tuple[str, ...] = ()
    residual_risks: tuple[str, ...] = ()
    review_reason: str = ""


@dataclass(frozen=True)
class ResolvedOrientationExcerpt:
    """One selected, bounded excerpt and its source provenance."""

    resource_id: str
    kind: str
    source_path: str
    section: str
    text: str
    score: int
    source_digest: str
    navigation_order: int

    @property
    def section_id(self) -> str:
        """Return the stable path-and-heading identity used in trace summaries."""
        return f"{self.source_path}#{self.section}"


@dataclass(frozen=True)
class ResolvedRoleOrientationPacket:
    """Rendered role card plus bounded content selected for one LLM call."""

    role: str
    orientation_identity: str
    context: str
    rendered: str
    excerpts: tuple[ResolvedOrientationExcerpt, ...]
    omitted_count: int
    content_digest: str

    def summary(self) -> dict[str, object]:
        """Return prompt-safe provenance without persisting source excerpts."""
        resource_ids: list[str] = []
        for excerpt in self.excerpts:
            if excerpt.resource_id not in resource_ids:
                resource_ids.append(excerpt.resource_id)
        return {
            "role": self.role,
            "orientation_identity": self.orientation_identity,
            "prompt_injected": True,
            "selected_resource_ids": resource_ids,
            "selected_sections": [excerpt.section_id for excerpt in self.excerpts],
            "context_chars": len(self.context),
            "omitted_count": self.omitted_count,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True)
class _DocumentSection:
    path: str
    heading: str
    text: str


def resolve_role_orientation_packet(
    role: str,
    query: RoleOrientationQuery,
    *,
    orientation: RoleOrientation | None = None,
    repo_root: str | Path | None = None,
    knowledge: Mapping[str, Any] | None = None,
    supplemental_context: str = "",
) -> ResolvedRoleOrientationPacket:
    """Resolve deterministic knowledge and documentation for one runtime role."""
    resolved_orientation = orientation or get_role_orientation(role)
    if resolved_orientation.role != role:
        raise ValueError(
            f"Orientation role {resolved_orientation.role!r} does not match {role!r}"
        )
    root = Path(repo_root or _REPO_ROOT).resolve()
    resolved_knowledge = (
        dict(knowledge)
        if knowledge is not None
        else _retrieve_knowledge_for_query(query)
    )
    candidates = _canonical_excerpts(
        resolved_orientation,
        query,
        resolved_knowledge,
        root=root,
    )
    candidates.extend(
        _documentation_excerpts(
            resolved_orientation,
            query,
            root=root,
        )
    )
    supplemental = _supplemental_excerpt(
        resolved_orientation,
        query,
        supplemental_context,
    )
    if supplemental is not None:
        candidates.append(supplemental)

    candidates.sort(
        key=lambda item: (
            item.navigation_order,
            -item.score,
            item.source_path,
            item.section,
        )
    )
    context, selected, omitted_count = _render_bounded_context(
        candidates,
        max_chars=resolved_orientation.max_context_chars,
        max_resource_chars=resolved_orientation.max_resource_chars,
    )
    card = (
        render_role_orientation_card(role)
        if orientation is None
        else _render_supplied_orientation_card(resolved_orientation)
    )
    rendered = f"{card}\n\n{context}" if context else card
    return ResolvedRoleOrientationPacket(
        role=role,
        orientation_identity=resolved_orientation.identity,
        context=context,
        rendered=rendered,
        excerpts=selected,
        omitted_count=omitted_count,
        content_digest=sha256(context.encode("utf-8")).hexdigest(),
    )


def _retrieve_knowledge_for_query(query: RoleOrientationQuery) -> dict[str, Any]:
    from trellis.agent.knowledge import get_store
    from trellis.agent.knowledge.schema import RetrievalSpec

    return get_store().retrieve_for_task(
        RetrievalSpec(
            method=query.method or None,
            features=list(query.features),
            instrument=query.instrument_type or None,
            model_family=query.model_family or None,
            candidate_engine_families=tuple(query.route_families),
            semantic_text_markers=tuple(_query_terms(query)),
            max_lessons=4,
        )
    )


def _canonical_excerpts(
    orientation: RoleOrientation,
    query: RoleOrientationQuery,
    knowledge: Mapping[str, Any],
    *,
    root: Path,
) -> list[ResolvedOrientationExcerpt]:
    excerpts: list[ResolvedOrientationExcerpt] = []
    for resource in orientation.navigation:
        if resource.kind in _RUNTIME_KINDS or resource.kind in _DOCUMENT_KINDS:
            continue
        text = ""
        section = resource.purpose
        if resource.resource_id == "product_decompositions":
            text = _render_decomposition(knowledge.get("decomposition"))
            text = _append_generated_skill_evidence(
                text,
                query,
                role=orientation.role,
            )
            text = _append_selected_principles(text, knowledge.get("principles", ()))
            section = "Product decomposition and principles"
        elif resource.resource_id == "admitted_routes":
            text = _render_route_identity(query)
            section = "Admitted route identity"
        elif resource.resource_id == "model_grammar":
            text = _render_model_grammar(knowledge.get("model_grammar", ()))
            section = "Selected model grammar"
        elif resource.resource_id == "method_requirements":
            text = _render_method_requirements(knowledge.get("method_requirements"))
            section = "Selected method requirements"
        elif resource.resource_id == "cookbook_catalog":
            text = _render_cookbook_evidence(
                knowledge.get("cookbook"),
                query,
                root=root,
                relative_path=resource.path,
            )
            if orientation.role == "model_validator":
                text = _append_generated_skill_evidence(
                    text,
                    query,
                    role=orientation.role,
                )
            section = "Read-only cookbook evidence"
        if not text.strip():
            continue
        prose = _sanitize_role_context(text)
        if not prose:
            continue
        excerpts.append(
            _make_excerpt(
                resource=resource,
                source_path=resource.path,
                section=section,
                text=prose,
                score=max(_score_text(query, section, prose), 1),
            )
        )
    return excerpts


def _render_decomposition(decomposition: Any) -> str:
    if decomposition is None:
        return ""
    lines = [
        f"Instrument: {getattr(decomposition, 'instrument', '')}",
        "Features: "
        + ", ".join(str(item) for item in getattr(decomposition, "features", ()) or ()),
        f"Preferred method: {getattr(decomposition, 'method', '')}",
    ]
    reasoning = str(getattr(decomposition, "reasoning", "") or "").strip()
    notes = str(getattr(decomposition, "notes", "") or "").strip()
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    if notes:
        lines.append(f"Notes: {notes}")
    return "\n".join(line for line in lines if not line.endswith(": "))


def _append_selected_principles(text: str, principles: Any) -> str:
    rows = []
    for principle in list(principles or ())[:3]:
        rule = str(getattr(principle, "rule", "") or "").strip()
        if rule:
            rows.append(f"Principle {getattr(principle, 'id', '')}: {rule}")
    if not rows:
        return text
    return "\n".join(part for part in (text, *rows) if part)


def _append_generated_skill_evidence(
    text: str,
    query: RoleOrientationQuery,
    *,
    role: str,
) -> str:
    """Append only role-safe records from the existing generated-skill index."""
    from trellis.agent.knowledge.skills import select_prompt_skill_artifacts

    audience = "routing" if role == "quant" else "review"
    stage = "route_selection" if role == "quant" else "model_validator_review"
    artifacts = select_prompt_skill_artifacts(
        "",
        audience=audience,
        stage=stage,
        instrument_type=query.instrument_type or None,
        pricing_method=query.method or None,
        route_ids=query.route_ids,
        route_families=query.route_families,
        knowledge_surface="distilled",
    )
    safe_kinds = {"principle", "lesson", "cookbook"}
    rows = []
    for artifact in artifacts:
        if artifact.get("kind") not in safe_kinds:
            continue
        summary = _sanitize_role_context(str(artifact.get("summary") or ""))
        if not summary:
            continue
        rows.append(
            f"Generated skill {artifact.get('id')}: {summary}"
        )
        if len(rows) >= 3:
            break
    if not rows:
        return text
    return "\n".join(part for part in (text, *rows) if part)


def _render_route_identity(query: RoleOrientationQuery) -> str:
    lines = []
    if query.route_ids:
        lines.append("Resolved route ids: " + ", ".join(query.route_ids))
    if query.route_families:
        lines.append("Admitted route families: " + ", ".join(query.route_families))
    return "\n".join(lines)


def _render_model_grammar(entries: Any) -> str:
    lines: list[str] = []
    for entry in list(entries or ())[:2]:
        title = str(getattr(entry, "title", "") or getattr(entry, "id", "")).strip()
        lines.append(title)
        model_name = str(getattr(entry, "model_name", "") or "").strip()
        if model_name:
            lines.append(f"Model: {model_name}")
        state = tuple(getattr(entry, "state_semantics", ()) or ())
        if state:
            lines.append("State semantics: " + "; ".join(str(item) for item in state))
        workflows = tuple(getattr(entry, "calibration_workflows", ()) or ())
        if workflows:
            lines.append("Calibration workflows: " + ", ".join(str(item) for item in workflows))
        deferred = tuple(getattr(entry, "deferred_scope", ()) or ())
        if deferred:
            lines.append("Deferred scope: " + ", ".join(str(item) for item in deferred))
        notes = str(getattr(entry, "notes", "") or "").strip()
        if notes:
            lines.append(notes)
    return "\n".join(lines)


def _render_method_requirements(requirements: Any) -> str:
    if requirements is None:
        return ""
    method = str(getattr(requirements, "method", "") or "").strip()
    rows = tuple(getattr(requirements, "requirements", ()) or ())
    lines = [f"Method: {method}"] if method else []
    lines.extend(f"- {row}" for row in rows[:6] if str(row).strip())
    return "\n".join(lines)


def _render_cookbook_evidence(
    cookbook: Any,
    query: RoleOrientationQuery,
    *,
    root: Path,
    relative_path: str,
) -> str:
    lines: list[str] = []
    if cookbook is not None:
        method = str(getattr(cookbook, "method", "") or "").strip()
        description = str(getattr(cookbook, "description", "") or "").strip()
        applicable = tuple(getattr(cookbook, "applicable_instruments", ()) or ())
        if method:
            lines.append(f"Method family: {method}")
        if description:
            lines.append(description)
        if applicable:
            lines.append("Applicable instruments: " + ", ".join(applicable))

    path = _confined_path(root, relative_path)
    if not path.exists():
        return "\n".join(lines)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, Mapping):
        return "\n".join(lines)
    method_key = query.method
    entry = data.get(method_key) if method_key else None
    if not isinstance(entry, Mapping):
        return "\n".join(lines)
    contracts = [
        contract
        for contract in entry.get("solution_contracts", ()) or ()
        if isinstance(contract, Mapping)
    ]
    if not contracts:
        return "\n".join(lines)
    contracts.sort(
        key=lambda contract: (
            -_score_text(
                query,
                str(contract.get("name") or contract.get("id") or ""),
                yaml.safe_dump(dict(contract), sort_keys=True),
            ),
            str(contract.get("id") or ""),
        )
    )
    selected = contracts[0]
    contract_score = _score_text(
        query,
        str(selected.get("name") or selected.get("id") or ""),
        yaml.safe_dump(dict(selected), sort_keys=True),
    )
    if contract_score <= 0:
        return "\n".join(lines)
    lines.append(
        "Solution contract: "
        + str(selected.get("name") or selected.get("id") or "")
    )
    assumptions = tuple(selected.get("assumptions", ()) or ())
    if assumptions:
        lines.append("Assumptions: " + "; ".join(str(item) for item in assumptions))
    payoff = str(selected.get("payoff") or "").strip()
    if payoff:
        lines.append(f"Payoff: {payoff}")
    market_data = tuple(selected.get("market_data", ()) or ())
    if market_data:
        lines.append("Market data: " + ", ".join(str(item) for item in market_data))
    return "\n".join(lines)


def _documentation_excerpts(
    orientation: RoleOrientation,
    query: RoleOrientationQuery,
    *,
    root: Path,
) -> list[ResolvedOrientationExcerpt]:
    excerpts: list[ResolvedOrientationExcerpt] = []
    for resource in orientation.navigation:
        if resource.kind not in _DOCUMENT_KINDS:
            continue
        sections = _sections_for_resource(resource, root=root)
        ranked: list[tuple[int, _DocumentSection]] = []
        for section in sections:
            prose = _sanitize_role_context(section.text)
            if not prose:
                continue
            score = _score_text(
                query,
                section.heading,
                prose[: orientation.max_resource_chars],
            )
            if score > 0:
                ranked.append((score, replace(section, text=prose)))
        if not ranked:
            continue
        ranked.sort(key=lambda item: (-item[0], item[1].path, item[1].heading))
        score, selected = ranked[0]
        excerpts.append(
            _make_excerpt(
                resource=resource,
                source_path=selected.path,
                section=selected.heading,
                text=selected.text,
                score=score,
            )
        )
    return excerpts


def _sections_for_resource(
    resource: OrientationResource,
    *,
    root: Path,
) -> list[_DocumentSection]:
    path = _confined_path(root, resource.path)
    if not path.exists() or not path.is_file():
        raise ValueError(
            f"Orientation resource {resource.resource_id!r} references missing {resource.path}"
        )
    source_paths = [(resource.path, path)]
    if resource.kind == "official_docs_index":
        source_paths.extend(_toctree_targets(path, resource.path, root=root))
    sections: list[_DocumentSection] = []
    seen: set[str] = set()
    for relative, source_path in source_paths:
        if relative in seen:
            continue
        seen.add(relative)
        text = source_path.read_text()
        sections.extend(_parse_document_sections(relative, text))
    return sections


def _toctree_targets(
    index_path: Path,
    index_relative: str,
    *,
    root: Path,
) -> list[tuple[str, Path]]:
    lines = index_path.read_text().splitlines()
    targets: list[tuple[str, Path]] = []
    in_toctree = False
    for line in lines:
        if line.strip().startswith(".. toctree::"):
            in_toctree = True
            continue
        if not in_toctree:
            continue
        if line and not line[0].isspace():
            in_toctree = False
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if "<" in stripped and stripped.endswith(">"):
            stripped = stripped.rsplit("<", 1)[1][:-1].strip()
        if "://" in stripped or stripped.startswith("/"):
            continue
        target = PurePosixPath(index_relative).parent / stripped
        target_candidates = (
            (target.with_suffix(".rst"), target.with_suffix(".md"))
            if target.suffix == ""
            else (target,)
        )
        resolved_target: tuple[str, Path] | None = None
        for candidate in target_candidates:
            relative = candidate.as_posix()
            target_path = _confined_path(root, relative)
            if target_path.exists() and target_path.is_file():
                resolved_target = (relative, target_path)
                break
        if resolved_target is None:
            raise ValueError(
                f"Documentation index {index_relative!r} references missing {stripped!r}"
            )
        _, resolved_path = resolved_target
        targets.append((resolved_path.relative_to(root).as_posix(), resolved_path))
    return targets


def _parse_document_sections(path: str, text: str) -> list[_DocumentSection]:
    if path.lower().endswith(".rst"):
        return _parse_rst_sections(path, text)
    return _parse_markdown_sections(path, text)


def _parse_rst_sections(path: str, text: str) -> list[_DocumentSection]:
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index in range(len(lines) - 1):
        title = lines[index].strip()
        underline = lines[index + 1].strip()
        if title and _RST_HEADING_UNDERLINE.fullmatch(underline):
            headings.append((index, title))
    return _materialize_sections(path, lines, headings, heading_span=2)


def _parse_markdown_sections(path: str, text: str) -> list[_DocumentSection]:
    lines = text.splitlines()
    headings = [
        (index, match.group(2).strip())
        for index, line in enumerate(lines)
        if (match := _MARKDOWN_HEADING.match(line)) is not None
    ]
    return _materialize_sections(path, lines, headings, heading_span=1)


def _materialize_sections(
    path: str,
    lines: list[str],
    headings: list[tuple[int, str]],
    *,
    heading_span: int,
) -> list[_DocumentSection]:
    if not headings:
        prose = "\n".join(lines).strip()
        return [_DocumentSection(path=path, heading=Path(path).stem, text=prose)] if prose else []
    sections: list[_DocumentSection] = []
    for position, (line_index, heading) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_index + heading_span : end]).strip()
        if body:
            sections.append(_DocumentSection(path=path, heading=heading, text=body))
    return sections


def _supplemental_excerpt(
    orientation: RoleOrientation,
    query: RoleOrientationQuery,
    text: str,
) -> ResolvedOrientationExcerpt | None:
    prose = _sanitize_role_context(text)
    if not prose:
        return None
    return ResolvedOrientationExcerpt(
        resource_id="shared_task_knowledge",
        kind="runtime_knowledge",
        source_path="runtime:shared_task_knowledge",
        section="Previously selected task knowledge",
        text=prose,
        score=max(_score_text(query, "Previously selected task knowledge", prose), 1),
        source_digest=sha256(prose.encode("utf-8")).hexdigest(),
        navigation_order=max(
            (
                item.order
                for item in orientation.navigation
                if item.kind in _RUNTIME_KINDS
            ),
            default=0,
        )
        + 1,
    )


def _make_excerpt(
    *,
    resource: OrientationResource,
    source_path: str,
    section: str,
    text: str,
    score: int,
) -> ResolvedOrientationExcerpt:
    return ResolvedOrientationExcerpt(
        resource_id=resource.resource_id,
        kind=resource.kind,
        source_path=source_path,
        section=section,
        text=text,
        score=score,
        source_digest=sha256(text.encode("utf-8")).hexdigest(),
        navigation_order=resource.order,
    )


def _render_bounded_context(
    candidates: list[ResolvedOrientationExcerpt],
    *,
    max_chars: int,
    max_resource_chars: int,
) -> tuple[str, tuple[ResolvedOrientationExcerpt, ...], int]:
    if not candidates or max_chars <= 0:
        return "", (), len(candidates)
    heading = "## Resolved Role Context\n"
    selected: list[ResolvedOrientationExcerpt] = []
    blocks: list[str] = []
    for candidate in candidates:
        excerpt_text, _ = _truncate(candidate.text, max_resource_chars)
        bounded = replace(candidate, text=excerpt_text)
        block = _render_excerpt(bounded)
        current = heading + "\n\n".join(blocks + [block])
        if len(current) <= max_chars:
            selected.append(bounded)
            blocks.append(block)
    omitted = len(candidates) - len(selected)
    context = heading + "\n\n".join(blocks)
    if omitted:
        while selected:
            omitted = len(candidates) - len(selected)
            marker = (
                f"[orientation context truncated; {omitted} excerpt(s) omitted]"
            )
            context = heading + "\n\n".join(
                _render_excerpt(excerpt) for excerpt in selected
            )
            overflow = len(context) + 2 + len(marker) - max_chars
            if overflow <= 0:
                context = f"{context}\n\n{marker}"
                break
            last = selected[-1]
            desired_text_chars = len(last.text) - overflow
            if desired_text_chars > 0:
                shortened, _ = _truncate(last.text, desired_text_chars)
                selected[-1] = replace(last, text=shortened)
                continue
            selected.pop()
        if not selected:
            omitted = len(candidates)
            marker = (
                f"[orientation context truncated; {omitted} excerpt(s) omitted]"
            )
            context, _ = _truncate(marker, max_chars)
    return context, tuple(selected), omitted


def _render_excerpt(excerpt: ResolvedOrientationExcerpt) -> str:
    return (
        f"### {excerpt.section}\n"
        f"Source: `{excerpt.section_id}` ({excerpt.resource_id})\n"
        f"{excerpt.text}"
    )


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 3:
        return text[:max_chars], True
    return text[: max_chars - 3].rstrip() + "...", True


def _score_text(query: RoleOrientationQuery, heading: str, body: str) -> int:
    terms = _query_terms(query)
    if not terms:
        return 1
    heading_tokens = set(_tokens(heading))
    body_tokens = set(_tokens(body))
    score = 0
    for term in terms:
        term_tokens = tuple(_tokens(term))
        if not term_tokens:
            continue
        if all(token in heading_tokens for token in term_tokens):
            score += 8 + len(term_tokens)
        elif all(token in body_tokens for token in term_tokens):
            score += 3 + len(term_tokens)
        else:
            score += 2 * len(heading_tokens.intersection(term_tokens))
            score += len(body_tokens.intersection(term_tokens))
    return score


def _query_terms(query: RoleOrientationQuery) -> tuple[str, ...]:
    raw = (
        query.instrument_type,
        query.method,
        query.model_family,
        *query.features,
        *query.route_ids,
        *query.route_families,
        *query.residual_risks,
        query.review_reason,
        query.description,
    )
    terms: list[str] = []
    for value in raw:
        normalized = " ".join(_tokens(str(value)))
        if normalized and normalized not in terms:
            terms.append(normalized)
    return tuple(terms)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _WORD.findall(text.lower().replace("_", " ").replace("-", " "))
        if token not in _STOP_WORDS and len(token) > 1
    )


def _strip_nonprose(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_fence = False
    rst_code_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith(".. code-block::") or stripped.startswith(".. code::"):
            rst_code_indent = len(line) - len(line.lstrip())
            continue
        if rst_code_indent is not None:
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if indent > rst_code_indent:
                continue
            rst_code_indent = None
        if re.match(r"^(from|import)\s+trellis(?:\.|\s)", stripped):
            continue
        output.append(line.rstrip())
    while output and not output[0].strip():
        output.pop(0)
    while output and not output[-1].strip():
        output.pop()
    compact: list[str] = []
    blank = False
    for line in output:
        if not line.strip():
            if not blank:
                compact.append("")
            blank = True
        else:
            compact.append(line)
            blank = False
    return "\n".join(compact).strip()


def _sanitize_role_context(text: str) -> str:
    """Remove code and builder-only construction authority from role packets."""
    prose = _strip_nonprose(text)
    safe_lines: list[str] = []
    for line in prose.splitlines():
        lowered = line.lower()
        if (
            "from trellis." in lowered
            or "import trellis" in lowered
            or " import " in lowered
            or "route_helper" in lowered
            or "route helper" in lowered
            or "helper-backed" in lowered
            or "checked helper" in lowered
            or "task-specific helper" in lowered
        ):
            continue
        safe_lines.append(line)
    return _strip_nonprose("\n".join(safe_lines))


def _confined_path(root: Path, relative_path: str) -> Path:
    if not relative_path or relative_path.startswith("runtime:"):
        raise ValueError(f"File-backed orientation resource requires a path: {relative_path!r}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Orientation resource path {relative_path!r} is outside repository root"
        ) from exc
    return candidate


def _render_supplied_orientation_card(orientation: RoleOrientation) -> str:
    lines = [
        f"## {orientation.title}",
        f"- Contract: `{orientation.identity}`",
        f"- Purpose: {orientation.purpose}",
        "### Owns",
        *(f"- {item}" for item in orientation.owns),
        "### Does Not Own",
        *(f"- {item}" for item in orientation.excludes),
        "### Navigation Order",
        *(
            f"{item.order}. [{item.kind}] `{item.path}`: {item.purpose}"
            for item in orientation.navigation
        ),
    ]
    return "\n".join(lines)
