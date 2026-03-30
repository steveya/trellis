# QUA-284 Arbitrary-Derivative Proving Run

## Summary
- Task id: `T998`
- Task title: `Himalaya ranked observation basket`
- Semantic contract: `ranked_observation_basket`
- Run seed: `20260328`
- Output file: `/Users/steveyang/Projects/steveya/trellis/docs/qua-284-arbitrary-derivative-proving-run.md`

## Scope and Caveats
- This is a proving run, not a full critic/arbiter/codegen build.
- Build payoff class: `HimalayaBasketPayoff`
- The knowledge system did not engage on this run; retrieval and promotion counts are zero.
- Comparison status: `None`
- The mock market data is intentionally simple: flat 20% equity vol, a generic correlation matrix, and no live provider wiring.

## Prompt
```text
Build a pricer for: Himalaya ranked observation basket

AAPL, MSFT, and NVDA with observation dates 2025-01-15, 2025-02-15, 2025-03-15. At each observation choose the best performer among remaining constituents, remove it, lock the simple return, and settle the average locked returns at maturity.
```

## Deterministic Decisions
```json
{
  "candidate_methods": [
    "monte_carlo"
  ],
  "preferred_method": "monte_carlo",
  "sample_indexing": {
    "kind": "path_index",
    "ordering": "simulation_generation_order",
    "start": 0
  },
  "sample_source": {
    "as_of": "2024-11-15",
    "kind": "market_snapshot",
    "snapshot_reference": {
      "as_of": "2024-11-15",
      "available_capabilities": [
        "black_vol_surface",
        "credit",
        "credit_curve",
        "discount",
        "discount_curve",
        "forward_curve",
        "forward_rate",
        "fx_rates",
        "jump_parameters",
        "local_vol_surface",
        "model_parameters",
        "spot",
        "state_space"
      ],
      "metadata": {
        "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
        "regime": "easing_cycle",
        "requested_as_of": "2024-11-15",
        "simulated": true,
        "snapshot_date": "2024-11-15"
      },
      "selected_components": {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "vol_surface": "usd_rates_smile"
      },
      "selected_curve_names": {
        "credit_curve": "usd_ig",
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M"
      },
      "source": "mock"
    },
    "source": "mock"
  },
  "semantic_contract_id": "ranked_observation_basket",
  "semantic_version": "c2.0",
  "simulation_seed": 14469743388754467197,
  "simulation_stream_id": "T998:61c308feb53f69db",
  "target_modules": [
    "trellis.models.resolution.basket_semantics",
    "trellis.models.monte_carlo.semantic_basket"
  ]
}
```

## Agent Decisions
```json
{
  "assembly_components": [
    "trellis.models.resolution.basket_semantics.resolve_basket_semantics",
    "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_process",
    "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_state_payoff",
    "trellis.models.monte_carlo.engine.MonteCarloEngine"
  ],
  "contract_surface": "ranked_observation_basket",
  "route_family": "family-name-free semantic basket route"
}
```

## Semantic Contract
```json
{
  "aggregation_rule": "average_locked_returns",
  "constituents": [
    "AAPL",
    "MSFT",
    "NVDA"
  ],
  "instrument_class": "basket_path_payoff",
  "lock_rule": "remove_selected",
  "observation_schedule": [
    "2025-01-15",
    "2025-02-15",
    "2025-03-15"
  ],
  "payoff_family": "basket_path_payoff",
  "payoff_rule": "ranked_observation_path_payoff",
  "primitive_families": [
    "correlated_basket_monte_carlo"
  ],
  "required_inputs": [
    "discount_curve",
    "underlier_spots",
    "black_vol_surface",
    "correlation_matrix"
  ],
  "selection_count": 1,
  "selection_operator": "best_of_remaining",
  "selection_scope": "remaining_constituents",
  "semantic_id": "ranked_observation_basket",
  "semantic_version": "c2.0",
  "settlement_rule": "settle_once_at_maturity",
  "target_modules": [
    "trellis.models.resolution.basket_semantics",
    "trellis.models.monte_carlo.semantic_basket"
  ],
  "underlier_structure": "multi_asset_basket"
}
```

## ProductIR Decomposition
```json
{
  "instrument_class": "basket_path_payoff",
  "multi_asset": true,
  "payoff_family": "basket_path_payoff",
  "payoff_rule": "ranked_observation_path_payoff",
  "required_inputs": [
    "discount_curve",
    "underlier_spots",
    "black_vol_surface",
    "correlation_matrix"
  ],
  "schedule_dependence": true,
  "settlement_rule": "settle_once_at_maturity",
  "state_dependence": "path_dependent",
  "underlier_structure": "multi_asset_basket"
}
```

## Semantic Trace
```json
{
  "market": {
    "as_of": "2024-11-15",
    "available_capabilities": [
      "black_vol_surface",
      "credit",
      "credit_curve",
      "discount",
      "discount_curve",
      "forward_curve",
      "forward_rate",
      "fx_rates",
      "jump_parameters",
      "local_vol_surface",
      "model_parameters",
      "spot",
      "state_space"
    ],
    "metadata": {
      "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
      "regime": "easing_cycle",
      "requested_as_of": "2024-11-15",
      "simulated": true,
      "snapshot_date": "2024-11-15"
    },
    "selected_components": {
      "discount_curve": "usd_ois",
      "forecast_curve": "USD-SOFR-3M",
      "vol_surface": "usd_rates_smile"
    },
    "selected_curve_names": {
      "credit_curve": "usd_ig",
      "discount_curve": "usd_ois",
      "forecast_curve": "USD-SOFR-3M"
    },
    "source": "mock"
  },
  "persisted_at": "2026-03-28T19:24:47.807589+00:00",
  "run_id": "20260328T152301130981",
  "summary": {
    "comparison_status": null,
    "deviations_pct": {},
    "error": null,
    "failures": [],
    "framework_outcome": null,
    "learning": {
      "captured_lesson_ids": [],
      "cookbook_candidate_paths": [],
      "cookbook_enriched": false,
      "knowledge_gap_log_paths": [],
      "knowledge_outcome": "no_new_knowledge",
      "knowledge_outcome_reason": "task succeeded without new reusable knowledge artifacts",
      "knowledge_trace_paths": [],
      "lessons_attributed": 7,
      "promotion_candidate_paths": [],
      "retrieved_lesson_ids": [
        "mc_017",
        "mc_020",
        "con_014",
        "mc_021",
        "mc_030",
        "con_015",
        "mc_031"
      ],
      "retrieved_lesson_titles": [
        "Ranked-observation basket must delegate to required semantic primitives",
        "Himalaya pricer must parse constituent names before basket resolution",
        "Bootstrap with pinned interpreter and explicit shell before pricing",
        "CorrelatedGBM requires 'mu' not 'mus' keyword argument",
        "RankedObservationBasketSpec does not accept start_date",
        "Do not import from trellis.conventions.schedule or day_count",
        "RankedObservationBasketSpec does not accept asset_names argument"
      ],
      "reusable_artifact_count": 0,
      "task_kind": "pricing"
    },
    "payoff_class": "HimalayaBasketPayoff",
    "preferred_method": null,
    "prices": {},
    "reference_target": null,
    "status": "succeeded",
    "success": true,
    "task_kind": "pricing",
    "token_usage": {
      "by_provider": {
        "anthropic": {
          "call_count": 6,
          "calls_with_usage": 6,
          "calls_without_usage": 0,
          "completion_tokens": 5170,
          "prompt_tokens": 43368,
          "total_tokens": 48538
        }
      },
      "by_stage": {
        "code_generation": {
          "call_count": 2,
          "calls_with_usage": 2,
          "calls_without_usage": 0,
          "completion_tokens": 2050,
          "prompt_tokens": 33573,
          "total_tokens": 35623
        },
        "critic": {
          "call_count": 1,
          "calls_with_usage": 1,
          "calls_without_usage": 0,
          "completion_tokens": 1494,
          "prompt_tokens": 4080,
          "total_tokens": 5574
        },
        "decomposition": {
          "call_count": 1,
          "calls_with_usage": 1,
          "calls_without_usage": 0,
          "completion_tokens": 963,
          "prompt_tokens": 4612,
          "total_tokens": 5575
        },
        "spec_design": {
          "call_count": 1,
          "calls_with_usage": 1,
          "calls_without_usage": 0,
          "completion_tokens": 462,
          "prompt_tokens": 811,
          "total_tokens": 1273
        },
        "unscoped": {
          "call_count": 1,
          "calls_with_usage": 1,
          "calls_without_usage": 0,
          "completion_tokens": 201,
          "prompt_tokens": 292,
          "total_tokens": 493
        }
      },
      "call_count": 6,
      "calls_with_usage": 6,
      "calls_without_usage": 0,
      "completion_tokens": 5170,
      "prompt_tokens": 43368,
      "total_tokens": 48538
    }
  },
  "task_id": "T998",
  "task_kind": "pricing",
  "workflow": {
    "active_trace_count": 0,
    "comparison_status": null,
    "latest_trace": {
      "action": "compile_only",
      "exists": true,
      "github_issue": null,
      "latest_event": "build_completed",
      "latest_event_details": {
        "attempts": 2
      },
      "latest_event_status": "ok",
      "linear_issue": null,
      "outcome": "build_completed",
      "path": "/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/traces/platform/executor_build_20260328192324_9c826be7.yaml",
      "request_id": "executor_build_20260328192324_9c826be7",
      "request_metadata": {
        "runtime_contract": {
          "description": "Build a pricer for: Himalaya ranked observation basket\n\nAAPL, MSFT, and NVDA with observation dates 2025-01-15, 2025-02-15, 2025-03-15. At each observation choose the best performer among remaining constituents, remove it, lock the simple return, and settle the average locked returns at maturity.",
          "evaluation_tags": [
            "task_runtime",
            "market:mock",
            "semantic_contract",
            "semantic:ranked_observation_basket"
          ],
          "instrument_type": "basket_option",
          "replay_key": "T998:61c308feb53f69db",
          "sample_indexing": {
            "kind": "path_index",
            "ordering": "simulation_generation_order",
            "start": 0
          },
          "sample_source": {
            "as_of": "2024-11-15",
            "kind": "market_snapshot",
            "snapshot_reference": {
              "as_of": "2024-11-15",
              "available_capabilities": [
                "black_vol_surface",
                "credit",
                "credit_curve",
                "discount",
                "discount_curve",
                "forward_curve",
                "forward_rate",
                "fx_rates",
                "jump_parameters",
                "local_vol_surface",
                "model_parameters",
                "spot",
                "state_space"
              ],
              "metadata": {
                "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
                "regime": "easing_cycle",
                "requested_as_of": "2024-11-15",
                "simulated": true,
                "snapshot_date": "2024-11-15"
              },
              "selected_components": {
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
                "vol_surface": "usd_rates_smile"
              },
              "selected_curve_names": {
                "credit_curve": "usd_ig",
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M"
              },
              "source": "mock"
            },
            "source": "mock"
          },
          "selected_curve_names": {
            "credit_curve": "usd_ig",
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M"
          },
          "semantic_contract": {
            "blueprint": {
              "primitive_families": [
                "correlated_basket_monte_carlo"
              ],
              "target_modules": [
                "trellis.models.resolution.basket_semantics",
                "trellis.models.monte_carlo.semantic_basket"
              ]
            },
            "market_data": {
              "optional_inputs": [],
              "required_inputs": [
                "discount_curve",
                "underlier_spots",
                "black_vol_surface",
                "correlation_matrix"
              ]
            },
            "methods": {
              "candidate_methods": [
                "monte_carlo"
              ],
              "preferred_method": "monte_carlo"
            },
            "product": {
              "aggregation_rule": "average_locked_returns",
              "constituents": [
                "AAPL",
                "MSFT",
                "NVDA"
              ],
              "exercise_style": "none",
              "instrument_class": "basket_path_payoff",
              "lock_rule": "remove_selected",
              "multi_asset": true,
              "observation_schedule": [
                "2025-01-15",
                "2025-02-15",
                "2025-03-15"
              ],
              "path_dependence": "path_dependent",
              "payoff_family": "basket_path_payoff",
              "payoff_rule": "ranked_observation_path_payoff",
              "schedule_dependence": true,
              "selection_count": 1,
              "selection_operator": "best_of_remaining",
              "selection_scope": "remaining_constituents",
              "settlement_rule": "settle_once_at_maturity",
              "state_dependence": "path_dependent",
              "underlier_structure": "multi_asset_basket"
            },
            "semantic_id": "ranked_observation_basket",
            "semantic_version": "c2.0"
          },
          "semantic_contract_id": "ranked_observation_basket",
          "simulation_identity": {
            "replay_key": "T998:61c308feb53f69db",
            "sample_indexing": {
              "kind": "path_index",
              "ordering": "simulation_generation_order",
              "start": 0
            },
            "sample_source": {
              "as_of": "2024-11-15",
              "kind": "market_snapshot",
              "snapshot_reference": {
                "as_of": "2024-11-15",
                "available_capabilities": [
                  "black_vol_surface",
                  "credit",
                  "credit_curve",
                  "discount",
                  "discount_curve",
                  "forward_curve",
                  "forward_rate",
                  "fx_rates",
                  "jump_parameters",
                  "local_vol_surface",
                  "model_parameters",
                  "spot",
                  "state_space"
                ],
                "metadata": {
                  "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
                  "regime": "easing_cycle",
                  "requested_as_of": "2024-11-15",
                  "simulated": true,
                  "snapshot_date": "2024-11-15"
                },
                "selected_components": {
                  "discount_curve": "usd_ois",
                  "forecast_curve": "USD-SOFR-3M",
                  "vol_surface": "usd_rates_smile"
                },
                "selected_curve_names": {
                  "credit_curve": "usd_ig",
                  "discount_curve": "usd_ois",
                  "forecast_curve": "USD-SOFR-3M"
                },
                "source": "mock"
              },
              "source": "mock"
            },
            "seed": 14469743388754467197,
            "seed_source": "derived_from_request_and_snapshot",
            "simulation_stream_id": "T998:61c308feb53f69db"
          },
          "simulation_seed": 14469743388754467197,
          "simulation_stream_id": "T998:61c308feb53f69db",
          "snapshot_reference": {
            "as_of": "2024-11-15",
            "available_capabilities": [
              "black_vol_surface",
              "credit",
              "credit_curve",
              "discount",
              "discount_curve",
              "forward_curve",
              "forward_rate",
              "fx_rates",
              "jump_parameters",
              "local_vol_surface",
              "model_parameters",
              "spot",
              "state_space"
            ],
            "metadata": {
              "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
              "regime": "easing_cycle",
              "requested_as_of": "2024-11-15",
              "simulated": true,
              "snapshot_date": "2024-11-15"
            },
            "selected_components": {
              "discount_curve": "usd_ois",
              "forecast_curve": "USD-SOFR-3M",
              "vol_surface": "usd_rates_smile"
            },
            "selected_curve_names": {
              "credit_curve": "usd_ig",
              "discount_curve": "usd_ois",
              "forecast_curve": "USD-SOFR-3M"
            },
            "source": "mock"
          },
          "task_id": "T998",
          "task_title": "Himalaya ranked observation basket"
        },
        "semantic_contract": {
          "blueprint": {
            "primitive_families": [
              "correlated_basket_monte_carlo"
            ],
            "target_modules": [
              "trellis.models.resolution.basket_semantics",
              "trellis.models.monte_carlo.semantic_basket"
            ]
          },
          "market_data": {
            "optional_inputs": [],
            "required_inputs": [
              "discount_curve",
              "underlier_spots",
              "black_vol_surface",
              "correlation_matrix"
            ]
          },
          "methods": {
            "candidate_methods": [
              "monte_carlo"
            ],
            "preferred_method": "monte_carlo"
          },
          "product": {
            "aggregation_rule": "average_locked_returns",
            "constituents": [
              "AAPL",
              "MSFT",
              "NVDA"
            ],
            "exercise_style": "none",
            "instrument_class": "basket_path_payoff",
            "lock_rule": "remove_selected",
            "multi_asset": true,
            "observation_schedule": [
              "2025-01-15",
              "2025-02-15",
              "2025-03-15"
            ],
            "path_dependence": "path_dependent",
            "payoff_family": "basket_path_payoff",
            "payoff_rule": "ranked_observation_path_payoff",
            "schedule_dependence": true,
            "selection_count": 1,
            "selection_operator": "best_of_remaining",
            "selection_scope": "remaining_constituents",
            "settlement_rule": "settle_once_at_maturity",
            "state_dependence": "path_dependent",
            "underlier_structure": "multi_asset_basket"
          },
          "semantic_id": "ranked_observation_basket",
          "semantic_version": "c2.0"
        },
        "task_id": "T998",
        "task_title": "Himalaya ranked observation basket"
      },
      "route_method": "monte_carlo",
      "selected_curve_names": {
        "credit_curve": "usd_ig",
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M"
      },
      "status": "succeeded",
      "token_usage": {
        "by_provider": {
          "anthropic": {
            "call_count": 6,
            "calls_with_usage": 6,
            "calls_without_usage": 0,
            "completion_tokens": 5170,
            "prompt_tokens": 43368,
            "total_tokens": 48538
          }
        },
        "by_stage": {
          "code_generation": {
            "call_count": 2,
            "calls_with_usage": 2,
            "calls_without_usage": 0,
            "completion_tokens": 2050,
            "prompt_tokens": 33573,
            "total_tokens": 35623
          },
          "critic": {
            "call_count": 1,
            "calls_with_usage": 1,
            "calls_without_usage": 0,
            "completion_tokens": 1494,
            "prompt_tokens": 4080,
            "total_tokens": 5574
          },
          "decomposition": {
            "call_count": 1,
            "calls_with_usage": 1,
            "calls_without_usage": 0,
            "completion_tokens": 963,
            "prompt_tokens": 4612,
            "total_tokens": 5575
          },
          "spec_design": {
            "call_count": 1,
            "calls_with_usage": 1,
            "calls_without_usage": 0,
            "completion_tokens": 462,
            "prompt_tokens": 811,
            "total_tokens": 1273
          },
          "unscoped": {
            "call_count": 1,
            "calls_with_usage": 1,
            "calls_without_usage": 0,
            "completion_tokens": 201,
            "prompt_tokens": 292,
            "total_tokens": 493
          }
        },
        "call_count": 6,
        "calls_with_usage": 6,
        "calls_without_usage": 0,
        "completion_tokens": 5170,
        "prompt_tokens": 43368,
        "total_tokens": 48538
      },
      "trace_kind": "platform",
      "updated_at": "2026-03-28T19:24:47.729273+00:00"
    },
    "linked_issues": {
      "github": [],
      "linear": []
    },
    "method_count": 0,
    "next_action": "Completed successfully.",
    "status": "succeeded"
  }
}
```

## Build Result
```json
{
  "comparison_status": null,
  "deviations_pct": {},
  "error": null,
  "failures": [],
  "framework_outcome": null,
  "learning": {
    "captured_lesson_ids": [],
    "cookbook_candidate_paths": [],
    "cookbook_enriched": false,
    "knowledge_gap_log_paths": [],
    "knowledge_outcome": "no_new_knowledge",
    "knowledge_outcome_reason": "task succeeded without new reusable knowledge artifacts",
    "knowledge_trace_paths": [],
    "lessons_attributed": 7,
    "promotion_candidate_paths": [],
    "retrieved_lesson_ids": [
      "mc_017",
      "mc_020",
      "con_014",
      "mc_021",
      "mc_030",
      "con_015",
      "mc_031"
    ],
    "retrieved_lesson_titles": [
      "Ranked-observation basket must delegate to required semantic primitives",
      "Himalaya pricer must parse constituent names before basket resolution",
      "Bootstrap with pinned interpreter and explicit shell before pricing",
      "CorrelatedGBM requires 'mu' not 'mus' keyword argument",
      "RankedObservationBasketSpec does not accept start_date",
      "Do not import from trellis.conventions.schedule or day_count",
      "RankedObservationBasketSpec does not accept asset_names argument"
    ],
    "reusable_artifact_count": 0,
    "task_kind": "pricing"
  },
  "payoff_class": "HimalayaBasketPayoff",
  "preferred_method": null,
  "prices": {},
  "reference_target": null,
  "status": "succeeded",
  "success": true,
  "task_kind": "pricing",
  "token_usage": {
    "by_provider": {
      "anthropic": {
        "call_count": 6,
        "calls_with_usage": 6,
        "calls_without_usage": 0,
        "completion_tokens": 5170,
        "prompt_tokens": 43368,
        "total_tokens": 48538
      }
    },
    "by_stage": {
      "code_generation": {
        "call_count": 2,
        "calls_with_usage": 2,
        "calls_without_usage": 0,
        "completion_tokens": 2050,
        "prompt_tokens": 33573,
        "total_tokens": 35623
      },
      "critic": {
        "call_count": 1,
        "calls_with_usage": 1,
        "calls_without_usage": 0,
        "completion_tokens": 1494,
        "prompt_tokens": 4080,
        "total_tokens": 5574
      },
      "decomposition": {
        "call_count": 1,
        "calls_with_usage": 1,
        "calls_without_usage": 0,
        "completion_tokens": 963,
        "prompt_tokens": 4612,
        "total_tokens": 5575
      },
      "spec_design": {
        "call_count": 1,
        "calls_with_usage": 1,
        "calls_without_usage": 0,
        "completion_tokens": 462,
        "prompt_tokens": 811,
        "total_tokens": 1273
      },
      "unscoped": {
        "call_count": 1,
        "calls_with_usage": 1,
        "calls_without_usage": 0,
        "completion_tokens": 201,
        "prompt_tokens": 292,
        "total_tokens": 493
      }
    },
    "call_count": 6,
    "calls_with_usage": 6,
    "calls_without_usage": 0,
    "completion_tokens": 5170,
    "prompt_tokens": 43368,
    "total_tokens": 48538
  }
}
```

## Build Observability
```json
{}
```

## Pricer Assembly
```json
{
  "adapter": "trellis.models.monte_carlo.semantic_basket.RankedObservationBasketMonteCarloPayoff",
  "engine": "trellis.models.monte_carlo.engine.MonteCarloEngine",
  "process_builder": "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_process",
  "resolver": "trellis.models.resolution.basket_semantics.resolve_basket_semantics",
  "state_payoff_builder": "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_state_payoff"
}
```

## Mock Pricing Run
```json
{
  "accrued_interest": 0.0,
  "clean_price": 2.4180013402,
  "dirty_price": 2.4180013402
}
```

## Final Price and Greeks
- clean_price: `2.4180013402`
- dirty_price: `2.4180013402`
- accrued_interest: `0`

- Spot deltas are zero in this setup because the payoff is expressed in relative-return units and is scale-invariant to the starting spot levels.

```json
{
  "common_vega": 13.6144365401,
  "correlation_sensitivity": 1.4511746643,
  "parallel_spot_delta": 0.0,
  "pricing_method": "monte_carlo",
  "pricing_note": "Common-random-number finite differences over the deterministic mock run.",
  "spot_deltas": {
    "AAPL": 0.0,
    "MSFT": 0.0,
    "NVDA": 0.0
  }
}
```

## Reproducibility
```json
{
  "correlation_matrix": [
    [
      1.0,
      0.35,
      0.25
    ],
    [
      0.35,
      1.0,
      0.3
    ],
    [
      0.25,
      0.3,
      1.0
    ]
  ],
  "correlation_preflight": {
    "correlation_status": "accepted",
    "max_asymmetry": 0.0,
    "max_diagonal_deviation": 0.0,
    "min_eigenvalue_after": 0.6413891153991625,
    "min_eigenvalue_before": 0.6413891153991627,
    "regularization_floor": 0.0,
    "requested_assets": 3,
    "source_key": "correlation_matrix",
    "source_kind": "explicit_matrix",
    "was_regularized": false
  },
  "discount_rate": 0.05,
  "expiry": "2025-03-15",
  "n_paths": 8192,
  "n_steps": 128,
  "notional": 100.0,
  "observation_dates": [
    "2025-01-15",
    "2025-02-15",
    "2025-03-15"
  ],
  "payoff_adapter": "RankedObservationBasketMonteCarloPayoff",
  "price_engine": "MonteCarloEngine",
  "resolved_correlation_matrix": [
    [
      1.0,
      0.35,
      0.25
    ],
    [
      0.35,
      1.0,
      0.3
    ],
    [
      0.25,
      0.3,
      1.0
    ]
  ],
  "resolved_time_to_expiry": 0.3287671233,
  "seed": 20260328,
  "settlement": "2024-11-15",
  "shock_shape": [
    8192,
    128,
    3
  ],
  "strike": 0.02,
  "underlier_spots": {
    "AAPL": 190.0,
    "MSFT": 410.0,
    "NVDA": 130.0
  },
  "vol": 0.2
}
```
