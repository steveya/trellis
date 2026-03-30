"""Reusable ranked-observation basket state helpers."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.event_state import (
    PathEventSpec,
    PathEventTimeline,
    build_event_path_requirement,
    event_step_indices,
    replay_path_event_timeline,
)
from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement, MonteCarloPathState

np = get_numpy()


def observation_step_indices(
    observation_times: tuple[float, ...],
    T: float,
    n_steps: int,
) -> tuple[int, ...]:
    """Map observation times onto deterministic Monte Carlo step indices."""
    return event_step_indices(observation_times, T, n_steps)


def build_basket_path_requirement(
    observation_times: tuple[float, ...],
    T: float,
    n_steps: int,
) -> MonteCarloPathRequirement:
    """Request the observation snapshots needed by ranked-observation basket payoffs."""
    return build_event_path_requirement(observation_times, T, n_steps)


def _basket_timeline(
    observation_steps: tuple[int, ...],
    *,
    selection_rule: str,
    lock_rule: str,
    aggregation_rule: str,
    selection_count: int,
) -> PathEventTimeline:
    events = [
        PathEventSpec(
            name=f"observation_{index}",
            kind="observation",
            step=step,
            payload={
                "selection_rule": selection_rule,
                "lock_rule": lock_rule,
                "aggregation_rule": aggregation_rule,
                "selection_count": selection_count,
            },
        )
        for index, step in enumerate(observation_steps, start=1)
    ]
    if observation_steps:
        events.append(
            PathEventSpec(
                name="basket_settlement",
                kind="settlement",
                step=observation_steps[-1],
                priority=1,
                payload={"rule": aggregation_rule},
            )
        )
    return PathEventTimeline(tuple(events))


def _settled_basket_value(state, aggregation_rule: str) -> np.ndarray:
    settlement = state.settlement_values.get("basket_settlement")
    if settlement is not None:
        return np.asarray(settlement, dtype=float)

    locked_returns = (
        np.zeros(state.n_paths, dtype=float)
        if state.locked_returns is None
        else np.asarray(state.locked_returns, dtype=float)
    )
    selected_counts = (
        np.zeros(state.n_paths, dtype=float)
        if state.selected_counts is None
        else np.asarray(state.selected_counts, dtype=float)
    )
    if aggregation_rule in {"average_locked_returns", "average_selected_returns"}:
        denominator = np.where(selected_counts > 0, selected_counts, 1.0)
        return locked_returns / denominator
    if aggregation_rule == "sum_locked_returns":
        return locked_returns
    raise ValueError(
        f"Unsupported basket aggregation_rule {aggregation_rule!r}; expected average_locked_returns or sum_locked_returns"
    )


def evaluate_ranked_observation_basket_paths(
    paths,
    initial_spots,
    observation_steps: tuple[int, ...],
    *,
    selection_rule: str,
    lock_rule: str,
    aggregation_rule: str,
    selection_count: int = 1,
):
    """Evaluate the basket payoff from a full Monte Carlo path tensor."""
    normalized = np.asarray(paths, dtype=float)
    if normalized.ndim == 2:
        normalized = normalized[:, :, np.newaxis]
    if normalized.ndim != 3:
        raise ValueError(
            f"Expected Monte Carlo paths with rank 2 or 3; received shape {normalized.shape}."
        )
    if not observation_steps:
        return np.zeros(0, dtype=float)

    observation_sections = tuple(normalized[:, step, :] for step in observation_steps)
    timeline = _basket_timeline(
        observation_steps,
        selection_rule=selection_rule,
        lock_rule=lock_rule,
        aggregation_rule=aggregation_rule,
        selection_count=selection_count,
    )
    sections = observation_sections + (observation_sections[-1],)
    state = replay_path_event_timeline(sections, initial_spots, timeline)
    return _settled_basket_value(state, aggregation_rule)


def evaluate_ranked_observation_basket_state(
    state: MonteCarloPathState,
    initial_spots,
    observation_steps: tuple[int, ...],
    *,
    selection_rule: str,
    lock_rule: str,
    aggregation_rule: str,
    selection_count: int = 1,
):
    """Evaluate the basket payoff from a reduced Monte Carlo path state."""
    if not observation_steps:
        return np.zeros(0, dtype=float)

    observation_sections = tuple(state.snapshot(step) for step in observation_steps)
    timeline = _basket_timeline(
        observation_steps,
        selection_rule=selection_rule,
        lock_rule=lock_rule,
        aggregation_rule=aggregation_rule,
        selection_count=selection_count,
    )
    sections = observation_sections + (observation_sections[-1],)
    replayed = replay_path_event_timeline(sections, initial_spots, timeline)
    return _settled_basket_value(replayed, aggregation_rule)


__all__ = [
    "build_basket_path_requirement",
    "evaluate_ranked_observation_basket_paths",
    "evaluate_ranked_observation_basket_state",
    "observation_step_indices",
]
