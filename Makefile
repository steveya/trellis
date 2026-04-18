PYTHON ?= /Users/steveyang/miniforge3/bin/python3
PYTEST ?= $(PYTHON) -m pytest
CANARY_FLAGS ?=
PR_CORE_MARKERS ?= not integration and not tier2
PR_GATE_IGNORES ?= --ignore=tests/test_contracts --ignore=tests/test_crossval --ignore=tests/test_verification --ignore=tests/test_tasks
PR_GATE_SHARD_COUNT ?= 4
PR_GATE_SHARD_INDEX ?= 1

.PHONY: gate-hygiene gate-pr-shard gate-tier2-contracts gate-pr gate-canary gate-release

gate-hygiene:
	$(PYTHON) scripts/test_hygiene.py --fail-on-ancient-unticketed-xfail

gate-tier2-contracts:
	$(PYTEST) tests/test_contracts/ -q -m "tier2 and not freshness"

gate-pr-shard:
	$(PYTHON) scripts/pr_gate_shard.py --count $(PR_GATE_SHARD_COUNT) --index $(PR_GATE_SHARD_INDEX) \
		| tr '\n' '\0' \
		| xargs -0 $(PYTEST) -x -q -m "$(PR_CORE_MARKERS)"

gate-pr:
	$(MAKE) gate-hygiene
	$(PYTEST) tests/ -x -q -m "$(PR_CORE_MARKERS)" $(PR_GATE_IGNORES)
	$(MAKE) gate-tier2-contracts

gate-canary:
	$(PYTHON) scripts/should_run_canary.py
	PYTHONHASHSEED=0 $(PYTHON) scripts/run_canary.py --subset core $(CANARY_FLAGS)

gate-release:
	$(MAKE) gate-pr
	$(PYTEST) tests/test_crossval tests/test_verification tests/test_tasks -x -q -m "not integration"
	$(PYTEST) tests/test_contracts/test_cassette_freshness.py -q -m "tier2 and freshness"
	PYTHONHASHSEED=0 $(PYTHON) scripts/run_canary.py --task T13 --replay --check-drift
	PYTHONHASHSEED=0 $(PYTHON) scripts/run_canary.py --task T38 --replay --check-drift
