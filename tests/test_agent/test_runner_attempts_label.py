"""Pin the corrected `attempts` console label semantics (QUA-874).

The benchmark runner used to print ``attempts=N`` for each task.  ``N``
counts calls to ``executor._generate_module``, i.e. LLM code-generation
attempts -- not retries.  A task that resolves through a deterministic
short-circuit prints ``attempts=0`` even when it succeeds.  The label
now reads ``llm_generation_attempts=N`` so future readers (and future
agents) cannot misread it as a retry count.
"""

from __future__ import annotations

import inspect

from trellis.agent import task_runtime


def test_run_task_console_line_uses_llm_generation_attempts_label():
    source = inspect.getsource(task_runtime.run_task)
    assert "llm_generation_attempts={result_data.get('attempts'" in source
    # The bare `attempts=` label was the misleading version; make sure the
    # f-string substring sits inside the longer `llm_generation_attempts=`
    # token rather than appearing standalone.
    assert ", attempts={result_data.get('attempts'" not in source
