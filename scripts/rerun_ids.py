"""Re-run specific task IDs."""

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

ids = set(sys.argv[1:])
all_tasks = load_tasks(status=None)
tasks = [task for task in all_tasks if task["id"] in ids]
ms = build_market_state()
output_file = ROOT / f"task_results_rerun_{datetime.now().strftime('%H%M')}.json"

results = []
for i, task in enumerate(tasks):
    print(f"[{i+1}/{len(tasks)}] {task['id']}: {task['title'][:50]}", flush=True)
    result = run_task(task, ms)
    results.append(result)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

ok = sum(1 for r in results if r.get("success"))
print(f"\n{'='*60}")
print(f"DONE: {ok}/{len(results)} succeeded → {output_file.name}")
