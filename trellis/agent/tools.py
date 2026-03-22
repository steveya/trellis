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
