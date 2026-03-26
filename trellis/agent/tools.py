"""Anthropic SDK tool definitions for the Trellis agent."""

from __future__ import annotations

TOOLS = [
    {
        "name": "inspect_library",
        "description": "Inspect the trellis library structure: list all modules, classes, and function signatures.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_module",
        "description": "Read the source code of a trellis module.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module_path": {
                    "type": "string",
                    "description": "Dotted module path, e.g. 'trellis.instruments.bond'",
                },
            },
            "required": ["module_path"],
        },
    },
    {
        "name": "find_symbol",
        "description": "Find where a public Trellis symbol is defined.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Public class or function name, e.g. 'theta_method_1d'",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "list_exports",
        "description": "List the public exports of a Trellis module.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module_path": {
                    "type": "string",
                    "description": "Dotted module path, e.g. 'trellis.models.pde.theta_method'",
                },
            },
            "required": ["module_path"],
        },
    },
    {
        "name": "resolve_import_candidates",
        "description": "Resolve one or more symbol names to candidate Trellis modules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Public symbol names to resolve.",
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "lookup_primitive_route",
        "description": "Build the deterministic primitive route plan for a product and method family.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural-language product description.",
                },
                "instrument_type": {
                    "type": "string",
                    "description": "Optional canonical instrument type, e.g. 'european_option'.",
                },
                "preferred_method": {
                    "type": "string",
                    "description": "Optional method family override, e.g. 'analytical' or 'monte_carlo'.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "build_thin_adapter_plan",
        "description": "Render the deterministic thin-adapter plan for a generated payoff.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural-language product description.",
                },
                "instrument_type": {
                    "type": "string",
                    "description": "Optional canonical instrument type.",
                },
                "preferred_method": {
                    "type": "string",
                    "description": "Optional method family override.",
                },
                "class_name": {
                    "type": "string",
                    "description": "Optional payoff class name for the adapter plan.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "select_invariant_pack",
        "description": "Select the deterministic invariant checks for one route/product family.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument_type": {
                    "type": "string",
                    "description": "Canonical instrument type.",
                },
                "method": {
                    "type": "string",
                    "description": "Method family, e.g. 'analytical'.",
                },
            },
            "required": ["method"],
        },
    },
    {
        "name": "build_comparison_harness",
        "description": "Build the deterministic comparison harness plan from TASK-style metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "object",
                    "description": "TASK-style object with construct and cross_validate fields.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "capture_cookbook_candidate",
        "description": "Extract a deterministic cookbook-candidate payload from successful generated code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method family for the candidate.",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of the successful build.",
                },
                "code": {
                    "type": "string",
                    "description": "Generated Python module source.",
                },
            },
            "required": ["method", "description", "code"],
        },
    },
    {
        "name": "search_repo",
        "description": "Search the Trellis package source tree for a text pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex or text to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_tests",
        "description": "Search Trellis tests for a text pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex or text to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_lessons",
        "description": "Search lessons and trace artifacts for a text pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex or text to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_module",
        "description": "Write or update a Python module in the trellis package.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path from the trellis package root, e.g. 'instruments/new_instrument.py'",
                },
                "content": {
                    "type": "string",
                    "description": "The full Python source code for the module.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run the trellis test suite (or a specific test file).",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Optional: specific test file or directory to run.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "fetch_market_data",
        "description": "Fetch Treasury yield data from FRED or Treasury.gov.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["fred", "treasury_gov"],
                    "description": "Data source to use.",
                },
                "as_of": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_pricing",
        "description": "Price an instrument using the trellis engine.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument_type": {
                    "type": "string",
                    "description": "Type of instrument, e.g. 'Bond'.",
                },
                "params": {
                    "type": "object",
                    "description": "Instrument constructor parameters.",
                },
                "curve_data": {
                    "type": "object",
                    "description": "Yield curve data: {tenor: rate} dict.",
                },
            },
            "required": ["instrument_type", "params", "curve_data"],
        },
    },
]
