"""Re-run specific task IDs."""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import load_env
load_env()
from trellis.agent.knowledge import reload
reload()

from trellis.agent.task_runtime import build_market_state, load_tasks, run_task

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("ids", nargs="+")
parser.add_argument(
    "--recovery-mode",
    choices=("strict", "assisted", "remediation"),
    default="assisted",
    help="Bounded automatic recovery mode for reruns.",
)
parser.add_argument("--validation", default="standard")
parser.add_argument(
    "--reuse",
    action="store_true",
    help="Reuse existing generated modules when present instead of forcing rebuilds.",
)
parser.add_argument(
    "--fresh-build",
    action="store_true",
    help=(
        "Bypass admitted adapter reuse and isolate the output path. This does not "
        "by itself require model-generated source."
    ),
)
parser.add_argument(
    "--generation-policy",
    choices=("deterministic_allowed", "builder_synthesis_required"),
    default="deterministic_allowed",
    help="Require observed builder-agent source synthesis or allow deterministic materialization.",
)
parser.add_argument(
    "--offline-local-agents",
    action="store_true",
    help=(
        "Forbid live LLM API calls during task execution. Deterministic quant, "
        "validation, exact bindings, and local/cassette paths may run; any "
        "attempted text or JSON LLM call fails the task."
    ),
)
args = parser.parse_args()

ids = set(args.ids)
all_tasks = load_tasks(status=None)
tasks = [task for task in all_tasks if task["id"] in ids]
ms = build_market_state()
output_file = ROOT / f"task_results_rerun_{datetime.now().strftime('%H%M')}.json"

results = []
for i, task in enumerate(tasks):
    print(f"[{i+1}/{len(tasks)}] {task['id']}: {task['title'][:50]}", flush=True)
    if args.offline_local_agents:
        from trellis.agent.offline_agents import offline_local_agent_run_scope

        with offline_local_agent_run_scope():
            result = run_task(
                task,
                ms,
                recovery_mode=args.recovery_mode,
                validation=args.validation,
                force_rebuild=(not args.reuse) or args.fresh_build,
                fresh_build=args.fresh_build,
                generation_policy=args.generation_policy,
                execution_mode_override="deterministic_replay",
            )
        result["offline_local_agents"] = True
    else:
        result = run_task(
            task,
            ms,
            recovery_mode=args.recovery_mode,
            validation=args.validation,
            force_rebuild=(not args.reuse) or args.fresh_build,
            fresh_build=args.fresh_build,
            generation_policy=args.generation_policy,
        )
    results.append(result)
    diagnosis_headline = result.get("task_diagnosis_headline")
    diagnosis_packet_path = result.get("task_diagnosis_packet_path")
    if diagnosis_headline:
        print(f"  diagnosis: {diagnosis_headline}", flush=True)
    if diagnosis_packet_path:
        print(f"  packet: {diagnosis_packet_path}", flush=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

ok = sum(1 for r in results if r.get("success"))
print(f"\n{'='*60}")
print(f"DONE: {ok}/{len(results)} succeeded → {output_file.name}")
