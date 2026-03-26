Hosting And Configuration
=========================

Trellis is currently hosted as a Python library, not as a standalone server in
this repository. In practice, "hosting" means choosing which long-running Python
process imports and executes the package.

Operational Modes
-----------------

The repo supports three practical modes:

- embedded library mode for notebooks, scripts, workers, or services that import ``trellis``
- task-runner mode via ``scripts/run_tasks.py`` and related batch tools
- experimental UI mode via the sibling ``trellis-ui`` repo, which uses this package as its pricing backend

There is no first-party HTTP application in this repo today, so production
deployments typically wrap Trellis inside another service boundary.

Installation Reality
--------------------

``setup.py`` currently packages the core library plus ``test`` and ``develop``
extras only. Optional runtime dependencies for agent workflows, live data, and
cross-validation are installed separately today.

.. code-block:: bash

   pip install trellis

   # Optional agent provider client
   pip install openai
   # or
   pip install anthropic

   # Optional live-data dependencies
   pip install requests fredapi

You may also need external libraries such as QuantLib, FinancePy, or other
comparison backends depending on which validation paths you intend to run.

Environment Loading
-------------------

``trellis.agent.config.load_env()`` searches upward for a project ``.env`` file
and loads values that are not already present in the process environment.

Key variables in the current codebase include:

- ``LLM_PROVIDER``: provider selector, currently ``openai`` or ``anthropic``
- ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``: provider credentials
- ``OPENAI_TEXT_TIMEOUT_SECONDS``, ``OPENAI_JSON_TIMEOUT_SECONDS``, ``OPENAI_MAX_RETRIES``: OpenAI request guards
- ``FRED_API_KEY``: optional live-data access for FRED
- ``GITHUB_REQUEST_AUDIT_TOKEN`` or ``GITHUB_TOKEN``: GitHub issue-sync auth
- ``GITHUB_REQUEST_AUDIT_REPOSITORY`` or ``GITHUB_REPOSITORY``: GitHub repo for request-audit issues
- ``GITHUB_REQUEST_AUDIT_LABELS``: optional GitHub labels for created issues
- ``LINEAR_API_KEY``: Linear issue-sync auth
- ``LINEAR_REQUEST_AUDIT_TEAM_ID`` or ``LINEAR_TEAM_ID``: Linear team key or id
- ``LINEAR_REQUEST_AUDIT_PROJECT_ID``: optional Linear project target
- ``NUMBA_CACHE_DIR``: recommended when running FinancePy-backed validation in restricted environments

Build And Docs
--------------

The public docs for this repo are built from ``docs/`` using Sphinx and
``myst-nb`` for notebook-backed tutorials.

.. code-block:: bash

   cd /Users/steveyang/Projects/steveya/trellis/docs
   make clean html

Related Reading
---------------

- :doc:`overview`
- :doc:`../validation/numba_cache`
- :doc:`../quant/index`

