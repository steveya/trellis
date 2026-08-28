from __future__ import annotations

import json
from pathlib import Path

import yaml


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    routes = {
        "routes": [
            {
                "id": "promoted_route",
                "status": "promoted",
                "primitives": [
                    {
                        "module": "trellis.models.example",
                        "symbol": "price_example",
                        "role": "route_helper",
                    },
                    {
                        "module": "trellis.models.example",
                        "symbol": "barrier_option_price",
                        "role": "route_helper",
                    },
                    {
                        "module": "trellis.models.example",
                        "symbol": "price_optional",
                        "role": "route_helper",
                        "required": False,
                    },
                ],
                "conditional_primitives": [
                    {
                        "when": {
                            "payoff_family": "example",
                            "methods": ["monte_carlo"],
                        },
                        "primitives": [
                            {
                                "module": "trellis.models.example",
                                "symbol": "price_example_monte_carlo",
                                "role": "route_helper",
                            }
                        ],
                    }
                ],
            },
            {
                "id": "candidate_route",
                "status": "candidate",
                "primitives": [
                    {
                        "module": "trellis.models.candidate",
                        "symbol": "price_candidate",
                        "role": "route_helper",
                    }
                ],
            },
        ]
    }
    bindings = {
        "bindings": [
            {
                "route_id": "promoted_route",
                "primitives": [
                    {
                        "module": "trellis.models.example",
                        "symbol": "price_example",
                        "role": "route_helper",
                    },
                    {
                        "module": "trellis.models.example",
                        "symbol": "barrier_option_price",
                        "role": "route_helper",
                    }
                ],
                "conditional_primitives": [
                    {
                        "when": "default",
                        "primitives": [
                            {
                                "module": "trellis.models.binding_only",
                                "symbol": "price_binding_only",
                                "role": "route_helper",
                            }
                        ],
                    }
                ],
            },
            {
                "route_id": "candidate_route",
                "primitives": [
                    {
                        "module": "trellis.models.candidate",
                        "symbol": "price_candidate",
                        "role": "route_helper",
                    }
                ],
            },
        ]
    }
    _write_yaml(
        tmp_path / "trellis/agent/knowledge/canonical/routes.yaml",
        routes,
    )
    _write_yaml(
        tmp_path / "trellis/agent/knowledge/canonical/backend_bindings.yaml",
        bindings,
    )
    adapter = tmp_path / "trellis/instruments/_agent/example.py"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text(
        """
from trellis.models.example import price_example as delegated_price
from trellis.models.example import barrier_option_price as delegated_barrier
from trellis.models.unused import price_unused
import trellis.models.binding_only as binding
import trellis.models.direct


def price_local():
    return 0.0


def evaluate():
    return delegated_price(None, None) + delegated_barrier(None, None) + binding.price_binding_only(None, None) + trellis.models.direct.price_direct(None, None) + price_local()
""".lstrip(),
        encoding="utf-8",
    )
    return tmp_path


def _fixture_root_with_indirect_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
from trellis.models.example import barrier_option_price


def accept_callback(callback):
    return callback


delegated_price = price_example


def evaluate():
    return accept_callback(barrier_option_price)
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_from_imported_module_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models import example


def accept_callback(callback):
    return callback


delegated_price = example.price_example


def callback_evaluate():
    return accept_callback(example.barrier_option_price)


def direct_evaluate():
    return example.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_imported_module_alias_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


delegated_module = helpers


def dynamic_evaluate():
    return getattr(helpers, "price_example")()


def delegated_evaluate():
    return delegated_module.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_same_named_non_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.other import price_example


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_relative_import_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from ...models.example import price_example
from ...models.example import barrier_option_price


delegated_barrier = barrier_option_price


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_authority_call_attribute(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


def evaluate():
    return helpers.price_example.__call__()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_nested_non_authority_shadow(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def shadowed():
    from trellis.models.other import price_example
    delegated = price_example
    return price_example()


def parameter_shadow(price_example):
    return price_example


delegated = price_example


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_nested_authority_shadow(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.other import price_example


def authoritative():
    from trellis.models.example import price_example
    delegated = price_example
    return price_example()


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_wildcard_authority_import(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import *


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_same_scope_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example as non_authority_last
from trellis.models.other import price_example as non_authority_last
from trellis.models.other import price_example as authority_last
from trellis.models.example import price_example as authority_last


non_authority_callback = non_authority_last
authority_callback = authority_last


def evaluate():
    return non_authority_last() + authority_last()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_late_global_authority_import(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def evaluate():
    global price_example
    return price_example()


from trellis.models.example import price_example
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_deferred_enclosing_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def evaluate():
    return price_example()


early_value = evaluate()
from trellis.models.other import price_example
late_value = evaluate()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_immediate_comprehension_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
early_values = [price_example() for _ in range(1)]
from trellis.models.other import price_example
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_authority_module_chain(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


def evaluate():
    return helpers.__dict__["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_authority_getattribute(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


def evaluate():
    return helpers.__getattribute__("price_example")()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_late_class_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def local_price():
    return 0.0


class Adapter:
    delegated = price_example()
    price_example = local_price
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_ordinary_rebindings(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
from trellis.models.example import barrier_option_price


def local_price():
    return 0.0


price_example = local_price


def barrier_option_price():
    return 0.0


def evaluate():
    return price_example() + barrier_option_price()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_assert_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

def local_price():
    return 0.0

assert (price_example := local_price)
value = price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_finally_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

def local_price():
    return 0.0

try:
    pass
finally:
    price_example = local_price
value = price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_first_context_manager_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from contextlib import nullcontext
from trellis.models.example import price_example

def local_price():
    return 0.0

with nullcontext(local_price) as price_example:
    pass
value = price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_annotation_only_references(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

price_example: object


class Adapter:
    price_example: object
    delegated = price_example()


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_deleted_class_binding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example as helper


def local_price():
    return 0.0


class Adapter:
    helper = local_price
    del helper
    delegated = helper()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_global_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def evaluate():
    return globals()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_local_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def evaluate():
    from trellis.models.example import price_example

    via_locals = locals()["price_example"]()
    via_vars = vars()["price_example"]()
    return via_locals + via_vars
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_aliased_dynamic_namespace_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


lookup = globals
via_alias = lookup()["price_example"]()
via_dunder = globals.__call__()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_unpacked_dynamic_namespace_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


(lookup,) = (globals,)
delegated = lookup()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_starred_unpacked_namespace_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


prefix_lookup, *_ = (globals,)
prefix_value = prefix_lookup()["price_example"]()
*_, suffix_lookup = (globals,)
suffix_value = suffix_lookup()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_shadowed_dynamic_namespace_imports(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import builtins
from trellis.models.example import price_example


def local_price():
    return 0.0


def local_namespace():
    return {}


namespace_lookup = globals
namespace_lookup = local_namespace
shadowed_alias_namespace = namespace_lookup()
globals = local_namespace
shadowed_builtin_namespace = globals()
price_example = local_price
module_namespace = builtins.globals()


def evaluate():
    from trellis.models.example import price_example as helper

    helper = local_price
    function_namespace = locals()
    return module_namespace, function_namespace
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_global_dynamic_namespace_import(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def evaluate():
    global price_example
    from trellis.models.example import price_example

    local_namespace = locals()
    delegated = globals()["price_example"]()
    return local_namespace, delegated
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_defaulted_dynamic_namespace_parameter(
    tmp_path: Path,
) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def evaluate(lookup=globals):
    return lookup()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_rebound_dynamic_namespace_parameter(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def local_namespace():
    return {}


def evaluate(lookup=globals):
    lookup = local_namespace
    return lookup()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_short_circuited_named_expression(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def local_price():
    return 0.0


False and (price_example := local_price)
delegated = price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_conditional_dynamic_namespace_alias(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def safe_lookup():
    return {}


enabled = True
lookup = globals if enabled else safe_lookup
delegated = lookup()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_comprehension_dynamic_namespace_alias(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

[lookup := globals for _ in (0,)]
delegated = lookup()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_deferred_generator_namespace_alias(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

def safe_lookup():
    return {}

stream = (lookup := globals for _ in (0,))
lookup = safe_lookup
next(stream)
delegated = lookup()["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_starred_dynamic_namespace_calls(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

delegated_positional = globals(*())["price_example"]()
delegated_keyword = globals(**{})["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_unconditional_control_flow_rebindings(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example as if_price
from trellis.models.example import price_example as while_price
from trellis.models.example import price_example as for_price
from trellis.models.example import price_example as match_price


def local_price():
    return 0.0


if (if_price := local_price):
    pass
if_value = if_price()

while (while_price := local_price):
    pass
while_value = while_price()

for _ in (for_price := ()):
    pass
for_value = for_price()

match (match_price := local_price):
    case _:
        pass
match_value = match_price()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_redirected_authority_imports(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def initialize_global():
    global global_price
    from trellis.models.example import price_example as global_price


initialize_global()
global_value = global_price()


def outer():
    def local_price():
        return 0.0

    nonlocal_price = local_price

    def initialize_nonlocal():
        nonlocal nonlocal_price
        from trellis.models.example import price_example as nonlocal_price

    initialize_nonlocal()
    return nonlocal_price()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_rebound_redirected_authority_imports(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def local_price():
    return 0.0

def initialize_global():
    global global_price
    from trellis.models.example import price_example as global_price

global_price = local_price
initialize_global()
global_value = global_price()

def outer():
    nonlocal_price = local_price

    def initialize_nonlocal():
        nonlocal nonlocal_price
        from trellis.models.example import price_example as nonlocal_price

    nonlocal_price = local_price
    initialize_nonlocal()
    return nonlocal_price()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_code_authority_use(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

evaluated = eval("price_example()")
executed = exec("price_example()")
runner = eval
aliased = runner("price_example()")


def safe_eval(source):
    return None


eval = safe_eval
safe = eval("price_example()")
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_code_authority_import(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
evaluated = eval("__import__('trellis.models.example')")
executed = exec("from trellis.models.example import price_example")
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_authority_imports(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
loaded = __import__("trellis.models.example", fromlist=["price_example"])
builtin_value = loaded.price_example()

import importlib
module_value = importlib.import_module("trellis.models.example").price_example()
loader = importlib.import_module
alias_value = loader("trellis.models.example").price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_container_selected_builtins(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
import importlib

builtin_loaded = (__import__,)[0]("trellis.models.example")
loaders = (importlib.import_module,)
module_loaded = loaders[0]("trellis.models.example")
namespace_value = (globals,)[0]()["price_example"]()
dynamic_value = (eval,)[0]("price_example()")
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_getattr_dynamic_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import importlib

loader = getattr(importlib, "import_module")
loaded = loader("trellis.models.example")
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_computed_getattr_dynamic_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import importlib

loader = getattr(importlib, "import_" + "module")
loaded = loader("trellis.models.example")
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_module_dict_dynamic_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import importlib

literal_loaded = importlib.__dict__["import_module"]("trellis.models.example")
loader_name = "import_module"
dynamic_loaded = importlib.__dict__[loader_name]("trellis.models.example")
literal_value = literal_loaded.price_example()
dynamic_value = dynamic_loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_vars_module_dynamic_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import importlib

literal_loaded = vars(importlib)["import_module"]("trellis.models.example")
loader_name = "import_module"
dynamic_loaded = vars(importlib)[loader_name]("trellis.models.example")
literal_value = literal_loaded.price_example()
dynamic_value = dynamic_loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_implicit_builtins_mapping(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
literal_loaded = __builtins__["__import__"](
    "trellis.models.example", fromlist=["price_example"]
)
mapping = __builtins__
loader_name = "__import__"
dynamic_loaded = mapping[loader_name](
    "trellis.models.example", fromlist=["price_example"]
)
literal_value = literal_loaded.price_example()
dynamic_value = dynamic_loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_global_builtins_mapping(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
loaded = globals()["__builtins__"]["__import__"](
    "trellis.models.example", fromlist=["price_example"]
)
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_shadowed_builtins_mapping(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
__builtins__ = {}
loaded = __builtins__["__import__"](
    "trellis.models.example", fromlist=["price_example"]
)
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_first_class_dangerous_builtins(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from importlib import import_module

loaded = next(map(import_module, ("trellis.models.example",)))
value = loaded.price_example()
source = ("__import__('trellis.models.example').price_example()",)
dynamic_value = next(map(eval, source))
def reflected_load(reflect):
    import importlib
    return reflect(importlib, "import_module")("trellis.models.example")
reflected = reflected_load(getattr)
reflected_value = reflected.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_aliased_dangerous_builtin_container(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def run(loaders):
    return loaders[0]("trellis.models.example", fromlist=["price_example"])
loaders = (__import__,)
value = run(loaders).price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_callable_returned_loaders(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from importlib import import_module

direct_loader = (lambda: __import__)()
direct_loaded = direct_loader("trellis.models.example", fromlist=["price_example"])
factory = lambda: import_module
named_loader = factory()
named_loaded = named_loader("trellis.models.example")
def choose_loader():
    return import_module
function_loader = choose_loader()
function_loaded = function_loader("trellis.models.example")
def choose_local_loader():
    from importlib import import_module as local_loader
    return local_loader
local_loader = choose_local_loader()
local_loaded = local_loader("trellis.models.example")
values = (
    direct_loaded.price_example(),
    named_loaded.price_example(),
    function_loaded.price_example(),
    local_loaded.price_example(),
)
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_generator_yielded_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def loaders():
    yield __import__
load = next(loaders())
loaded = load("trellis.models.example", fromlist=["price_example"])
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_generator_yield_from_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def loaders():
    yield from (__import__,)
load = next(loaders())
loaded = load("trellis.models.example", fromlist=["price_example"])
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_async_callable_returned_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
async def get_loader():
    return __import__

async def evaluate():
    loader = await get_loader()
    loaded = loader("trellis.models.example", fromlist=["price_example"])
    return loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_method_returned_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
class Factory:
    @staticmethod
    def loader():
        return __import__

load = Factory.loader()
loaded = load("trellis.models.example", fromlist=["price_example"])
value = loaded.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_object_getattribute_loader(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import importlib
from builtins import object as base_object

loader = object.__getattribute__(importlib, "import_module")
loaded = loader("trellis.models.example")
aliased_loader = base_object.__getattribute__(importlib, "import_module")
aliased_loaded = aliased_loader("trellis.models.example")
values = (loaded.price_example(), aliased_loaded.price_example())
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_module_getattribute_loaders(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import importlib

direct_loader = importlib.__getattribute__("import_module")
direct_loaded = direct_loader("trellis.models.example")
getter = importlib.__getattribute__
aliased_loader = getter("import_module")
aliased_loaded = aliased_loader("trellis.models.example")
values = (direct_loaded.price_example(), aliased_loaded.price_example())
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_first_class_namespace_arguments(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
def run(lookup):
    return lookup()["price_example"]()
run(globals)
alias = globals
run(alias)
run(globals())
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_callee_local_namespace_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def run(lookup):
    from trellis.models.example import price_example
    return lookup()["price_example"]()

run(locals)
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_current_module_vars_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
import sys as runtime

direct_namespace = vars(runtime.modules[__name__])
direct_value = direct_namespace["price_example"]()
current_module = runtime.modules[__name__]
aliased_namespace = vars(current_module)
aliased_value = aliased_namespace["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_sys_modules_authority_lookup(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import sys
import trellis.models.example as helpers

helpers = None
direct_value = sys.modules["trellis.models.example"].price_example()
get_value = sys.modules.get("trellis.models.example").price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_definition_expression_rebindings(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
from trellis.models.example import price_example as decorated_price
from trellis.models.example import price_example as lambda_price

def local_price():
    return 0.0

def configured(value=(price_example := local_price)):
    return value

default_value = price_example()

@(decorated_price := (lambda function: function))
def decorated():
    return 0.0

decorated_value = decorated_price(decorated)
runner = lambda value=(lambda_price := local_price): value
lambda_value = lambda_price()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def test_audit_preserves_required_route_and_binding_authority_drift(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(_fixture_root(tmp_path))

    assert report.promoted_route_count == 1
    assert [(item.route_id, item.condition, item.symbol) for item in report.route_authority] == [
        ("promoted_route", "base", "barrier_option_price"),
        ("promoted_route", "base", "price_example"),
        (
            "promoted_route",
            '{"methods":["monte_carlo"],"payoff_family":"example"}',
            "price_example_monte_carlo",
        ),
    ]
    assert [(item.route_id, item.condition, item.symbol) for item in report.binding_authority] == [
        ("promoted_route", "base", "barrier_option_price"),
        ("promoted_route", "base", "price_example"),
        ("promoted_route", '"default"', "price_binding_only"),
    ]
    assert [item.symbol for item in report.route_only_authority] == [
        "price_example_monte_carlo"
    ]
    assert [item.symbol for item in report.binding_only_authority] == [
        "price_binding_only"
    ]


def test_audit_resolves_import_aliases_and_ignores_unused_or_local_price_calls(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(_fixture_root(tmp_path))

    assert [item.symbol for item in report.adapter_calls] == [
        "price_binding_only",
        "barrier_option_price",
        "price_example",
        "price_direct",
    ]
    assert all(
        item.matches_required_authority
        for item in report.adapter_calls
        if item.symbol != "price_direct"
    )
    direct = next(item for item in report.adapter_calls if item.symbol == "price_direct")
    assert direct.module == "trellis.models.direct"
    assert direct.matches_required_authority is False
    assert [item.symbol for item in report.adapter_calls if item.is_price_call] == [
        "price_binding_only",
        "price_example",
        "price_direct",
    ]
    example = next(item for item in report.adapter_calls if item.symbol == "price_example")
    assert example.local_name == "delegated_price"
    assert example.path == "trellis/instruments/_agent/example.py"
    assert example.line == 13


def test_helper_authority_report_has_stable_machine_readable_shape(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    payload = build_helper_authority_report(_fixture_root(tmp_path)).to_dict()

    assert payload["schema_version"] == 2
    assert payload["summary"] == {
        "promoted_route_count": 1,
        "route_authority_route_count": 1,
        "route_authority_reference_count": 3,
        "binding_authority_route_count": 1,
        "binding_authority_reference_count": 3,
        "route_only_reference_count": 1,
        "binding_only_reference_count": 1,
        "adapter_price_call_file_count": 1,
        "adapter_price_call_count": 3,
        "adapter_authority_call_file_count": 1,
        "adapter_authority_call_count": 3,
        "adapter_indirect_authority_use_file_count": 0,
        "adapter_indirect_authority_use_count": 0,
    }
    assert payload["adapter_indirect_authority_uses"] == []
    assert json.loads(json.dumps(payload)) == payload


def test_audit_rejects_assignment_aliases_and_callback_authority(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_indirect_authority(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (9, "price_example", "price_example", "indirect_reference"),
        (
            13,
            "barrier_option_price",
            "barrier_option_price",
            "indirect_reference",
        ),
    ]
    assert report.has_adapter_authority is True
    assert report.summary["adapter_indirect_authority_use_file_count"] == 1
    assert report.summary["adapter_indirect_authority_use_count"] == 2


def test_audit_resolves_authority_through_from_imported_modules(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_from_imported_module_authority(tmp_path)
    )

    assert [
        (item.line, item.local_name, item.module, item.symbol)
        for item in report.adapter_calls
    ] == [
        (
            16,
            "example.price_example",
            "trellis.models.example",
            "price_example",
        )
    ]
    assert [
        (item.line, item.local_name, item.module, item.symbol)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            8,
            "example.price_example",
            "trellis.models.example",
            "price_example",
        ),
        (
            12,
            "example.barrier_option_price",
            "trellis.models.example",
            "barrier_option_price",
        ),
    ]
    assert report.has_adapter_authority is True
    assert report.summary["adapter_indirect_authority_use_file_count"] == 1
    assert report.summary["adapter_indirect_authority_use_count"] == 2


def test_audit_rejects_imported_authority_modules_used_as_values(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_imported_module_alias_authority(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (
            item.line,
            item.local_name,
            item.module,
            item.symbol,
            item.use_kind,
        )
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        ),
        (
            8,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_does_not_match_same_named_symbol_from_other_module(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_same_named_non_authority(tmp_path)
    )

    assert len(report.adapter_calls) == 1
    assert report.adapter_calls[0].module == "trellis.models.other"
    assert report.adapter_calls[0].symbol == "price_example"
    assert report.adapter_calls[0].is_price_call is True
    assert report.adapter_calls[0].matches_required_authority is False
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_normalizes_relative_authority_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_relative_import_authority(tmp_path)
    )

    assert [
        (item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [("trellis.models.example", "price_example", True)]
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "barrier_option_price",
            "indirect_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_rejects_authority_reached_through_call_attribute(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_authority_call_attribute(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_keeps_nested_non_authority_import_scoped(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_nested_non_authority_shadow(tmp_path)
    )

    assert [
        (item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        ("trellis.models.other", "price_example", False),
        ("trellis.models.example", "price_example", True),
    ]
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]


def test_audit_keeps_nested_authority_import_scoped(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_nested_authority_shadow(tmp_path)
    )

    assert [
        (item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        ("trellis.models.example", "price_example", True),
        ("trellis.models.other", "price_example", False),
    ]
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]


def test_audit_rejects_wildcard_imports_from_authority_modules(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_wildcard_authority_import(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            1,
            "*",
            "trellis.models.example",
            "*",
            "wildcard_import",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_respects_same_scope_import_rebinding_order(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_same_scope_rebinding(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        ("authority_last", "trellis.models.example", True),
        ("non_authority_last", "trellis.models.other", False),
    ]
    assert [
        (item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "authority_last",
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]


def test_audit_resolves_late_module_import_for_explicit_global(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_late_global_authority_import(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            "price_example",
            "trellis.models.example",
            "price_example",
            True,
        )
    ]
    assert report.adapter_indirect_authority_uses == ()


def test_audit_retains_possible_enclosing_imports_for_deferred_calls(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_deferred_enclosing_rebinding(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            "price_example",
            "trellis.models.example",
            "price_example",
            True,
        ),
        (
            "price_example",
            "trellis.models.other",
            "price_example",
            False,
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_uses_creation_position_for_immediate_comprehensions(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_immediate_comprehension_rebinding(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            "price_example",
            "trellis.models.example",
            "price_example",
            True,
        )
    ]


def test_audit_retains_authority_module_root_in_dynamic_attribute_chain(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_authority_module_chain(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_retains_authority_module_root_for_dynamic_getattribute(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_authority_getattribute(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_early_class_reference_before_late_rebinding(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_late_class_rebinding(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [(9, "trellis.models.example", "price_example", True)]
    assert report.has_adapter_authority is True


def test_audit_honors_ordinary_rebindings_after_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_ordinary_rebindings(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_treats_assert_walrus_rebindings_as_conditional(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_assert_rebinding(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [(7, "trellis.models.example", "price_example", True)]
    assert report.has_adapter_authority is True


def test_audit_treats_finally_rebindings_as_unconditional(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_finally_rebinding(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_treats_first_context_manager_target_as_unconditional(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_first_context_manager_rebinding(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_preserves_imports_across_annotation_only_statements(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_annotation_only_references(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (8, "trellis.models.example", "price_example", True),
        (12, "trellis.models.example", "price_example", True),
    ]
    assert report.has_adapter_authority is True


def test_audit_restores_outer_lookup_after_deleted_class_binding(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_deleted_class_binding(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [(11, "trellis.models.example", "price_example", True)]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_for_dynamic_global_namespace_lookup(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_global_lookup(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_for_dynamic_local_namespace_lookup(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_local_lookup(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_local_namespace",
        ),
        (
            5,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_local_namespace",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_for_aliased_dynamic_namespace_lookup(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_aliased_dynamic_namespace_lookup(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
        (
            6,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_unpacked_dynamic_namespace_aliases(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_unpacked_dynamic_namespace_lookup(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_starred_unpacked_dynamic_namespace_aliases(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_starred_unpacked_namespace_lookup(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
        (
            7,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_filters_dynamic_namespaces_to_active_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_shadowed_dynamic_namespace_imports(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_places_global_imports_only_in_the_global_namespace(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_global_dynamic_namespace_import(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            6,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_defaulted_dynamic_namespace_parameters(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_defaulted_dynamic_namespace_parameter(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_honors_rebinding_after_dynamic_namespace_parameter_defaults(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_rebound_dynamic_namespace_parameter(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_keeps_imports_shadowed_only_by_short_circuited_named_expressions(
    tmp_path,
):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_short_circuited_named_expression(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            9,
            "trellis.models.example",
            "price_example",
            True,
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_conditional_dynamic_namespace_aliases(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_conditional_dynamic_namespace_alias(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            10,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_comprehension_dynamic_namespace_aliases(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_comprehension_dynamic_namespace_alias(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_preserves_deferred_generator_namespace_aliases(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_deferred_generator_namespace_alias(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            9,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_treats_starred_namespace_calls_as_potentially_empty(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_starred_dynamic_namespace_calls(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            3,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
        (
            4,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_treats_control_flow_headers_as_unconditional_rebindings(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_unconditional_control_flow_rebindings(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_propagates_global_and_nonlocal_authority_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_redirected_authority_imports(tmp_path)
    )

    assert [
        (item.line, item.local_name, item.module, item.symbol)
        for item in report.adapter_calls
    ] == [
        (
            7,
            "global_price",
            "trellis.models.example",
            "price_example",
        ),
        (
            21,
            "nonlocal_price",
            "trellis.models.example",
            "price_example",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_preserves_redirected_imports_after_owner_rebindings(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_rebound_redirected_authority_imports(tmp_path)
    )

    assert [
        (item.line, item.local_name, item.module, item.symbol)
        for item in report.adapter_calls
    ] == [
        (
            10,
            "global_price",
            "trellis.models.example",
            "price_example",
        ),
        (
            21,
            "nonlocal_price",
            "trellis.models.example",
            "price_example",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_on_dynamic_code_with_active_authority(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_code_authority_use(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            3,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_code_eval",
        ),
        (
            4,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_code_exec",
        ),
        (
            6,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_code_eval",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_on_authority_imported_by_dynamic_code(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_code_authority_import(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (1, "eval", "*", "*", "dynamic_code_eval"),
        (2, "exec", "*", "*", "dynamic_code_exec"),
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_on_dynamic_authority_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_authority_imports(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            1,
            "__import__",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
        (
            5,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
        (
            7,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_builtins_selected_through_containers(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_container_selected_builtins(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "__import__",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
        (
            6,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
        (
            7,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
        (
            8,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_code_eval",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dynamic_loaders_reached_through_getattr(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_getattr_dynamic_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_for_computed_getattr_loader_names(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_computed_getattr_dynamic_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dynamic_loaders_reached_through_module_dict(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_module_dict_dynamic_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            3,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
        (
            5,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dynamic_loaders_reached_through_vars_module(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_vars_module_dynamic_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            3,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
        (
            5,
            "import_module",
            "trellis.models.example",
            "*",
            "dynamic_import",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dynamic_loaders_from_implicit_builtins_mapping(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_implicit_builtins_mapping(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (1, "__import__", "trellis.models.example", "*", "dynamic_import"),
        (6, "__import__", "trellis.models.example", "*", "dynamic_import"),
        (6, "eval", "*", "*", "dynamic_code_eval"),
        (6, "exec", "*", "*", "dynamic_code_exec"),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_builtins_reached_through_global_namespace_mapping(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_global_builtins_mapping(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (1, "__import__", "trellis.models.example", "*", "dynamic_import")
    ]
    assert report.has_adapter_authority is True


def test_audit_respects_shadowed_implicit_builtins_mapping(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_shadowed_builtins_mapping(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_fails_closed_on_first_class_dangerous_builtins(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_first_class_dangerous_builtins(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (3, "import_module", "*", "*", "first_class_dynamic_import"),
        (6, "eval", "*", "*", "first_class_dynamic_code_eval"),
        (10, "getattr", "*", "*", "first_class_reflection_getattr"),
    ]
    assert report.has_adapter_authority is True


def test_audit_follows_dangerous_builtins_through_aliased_containers(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_aliased_dangerous_builtin_container(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (4, "__import__", "*", "*", "first_class_dynamic_import")
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dangerous_builtins_returned_by_callables(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_callable_returned_loaders(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (4, "__import__", "trellis.models.example", "*", "dynamic_import"),
        (7, "import_module", "trellis.models.example", "*", "dynamic_import"),
        (11, "import_module", "trellis.models.example", "*", "dynamic_import"),
        (16, "import_module", "trellis.models.example", "*", "dynamic_import"),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dangerous_builtins_yielded_by_generators(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_generator_yielded_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (3, "__import__", "*", "*", "first_class_dynamic_import"),
        (4, "__import__", "trellis.models.example", "*", "dynamic_import"),
    ]
    assert report.has_adapter_authority is True


def test_audit_expands_dangerous_builtins_yielded_from_containers(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_generator_yield_from_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (3, "__import__", "*", "*", "first_class_dynamic_import"),
        (4, "__import__", "trellis.models.example", "*", "dynamic_import"),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dangerous_builtins_returned_by_async_callables(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_async_callable_returned_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (6, "__import__", "trellis.models.example", "*", "dynamic_import")
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_dangerous_builtins_returned_by_methods(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_method_returned_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (7, "__import__", "trellis.models.example", "*", "dynamic_import")
    ]
    assert report.has_adapter_authority is True


def test_audit_recognizes_object_getattribute_as_reflection(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_object_getattribute_loader(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (5, "import_module", "trellis.models.example", "*", "dynamic_import"),
        (7, "import_module", "trellis.models.example", "*", "dynamic_import"),
    ]
    assert report.has_adapter_authority is True


def test_audit_recognizes_module_bound_getattribute_as_reflection(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_module_getattribute_loaders(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (4, "import_module", "trellis.models.example", "*", "dynamic_import"),
        (7, "import_module", "trellis.models.example", "*", "dynamic_import"),
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_on_first_class_namespace_arguments(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_first_class_namespace_arguments(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "price_example",
            "trellis.models.example",
            "price_example",
            "first_class_global_namespace",
        ),
        (
            6,
            "price_example",
            "trellis.models.example",
            "price_example",
            "first_class_global_namespace",
        ),
        (
            7,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_on_first_class_namespace_transfer(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_callee_local_namespace_authority(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (5, "local", "*", "*", "first_class_local_namespace")
    ]
    assert report.has_adapter_authority is True


def test_audit_fails_closed_for_vars_of_current_module(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_current_module_vars_lookup(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
        (
            7,
            "price_example",
            "trellis.models.example",
            "price_example",
            "dynamic_global_namespace",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_authority_modules_recovered_from_sys_modules(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_sys_modules_authority_lookup(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (5, "trellis.models.example", "price_example", True),
        (6, "trellis.models.example", "price_example", True),
    ]
    assert report.has_adapter_authority is True


def test_audit_collects_definition_expression_rebindings(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_definition_expression_rebindings(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_helper_authority_human_report_surfaces_drift_and_adapter_authority(tmp_path):
    from trellis.agent.helper_authority_audit import (
        build_helper_authority_report,
        render_helper_authority_report,
    )

    rendered = render_helper_authority_report(
        build_helper_authority_report(_fixture_root(tmp_path))
    )

    assert "Helper authority audit" in rendered
    assert "route_authority_references=3" in rendered
    assert "binding_authority_references=3" in rendered
    assert "route_only_references=1" in rendered
    assert "binding_only_references=1" in rendered
    assert "adapter_indirect_authority_uses=0" in rendered
    assert "price_example_monte_carlo" in rendered
    assert "price_binding_only" in rendered
    assert "barrier_option_price" in rendered
    assert "trellis/instruments/_agent/example.py:13" in rendered


def test_current_repository_helper_authority_report_is_internally_consistent():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)

    assert report.promoted_route_count > 0
    assert all(item.required for item in report.route_authority)
    assert all(item.required for item in report.binding_authority)
    assert all((root / item.path).is_file() for item in report.adapter_calls)
    assert all(
        (root / item.path).is_file()
        for item in report.adapter_indirect_authority_uses
    )
    assert report.to_dict()["summary"]["route_authority_reference_count"] == len(
        report.route_authority
    )


def test_current_repository_has_zero_admitted_adapter_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    authority_calls = [
        item for item in report.adapter_calls if item.matches_required_authority
    ]

    assert authority_calls == []
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False
    assert report.summary["adapter_authority_call_file_count"] == 0
    assert report.summary["adapter_authority_call_count"] == 0
    assert report.summary["adapter_indirect_authority_use_file_count"] == 0
    assert report.summary["adapter_indirect_authority_use_count"] == 0


def test_current_repository_retires_arithmetic_asian_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    asian_symbols = {
        "price_asian_option_monte_carlo",
        "price_arithmetic_asian_option_analytical",
        "price_arithmetic_asian_option_monte_carlo",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in asian_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/asianoption.py"
        and item.symbol in asian_symbols
    ]


def test_current_repository_retires_single_name_cds_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    retired_symbols = {
        "build_cds_schedule",
        "price_cds_analytical",
        "price_cds_monte_carlo",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in retired_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/cds.py"
        and item.symbol in retired_symbols
    ]
    summary = report.to_dict()["summary"]
    assert summary["route_authority_reference_count"] <= 39
    assert summary["binding_authority_reference_count"] <= 43
    assert summary["route_only_reference_count"] <= 2
    assert summary["binding_only_reference_count"] <= 6


def test_current_repository_classifies_scalar_barrier_formula_as_pricing_kernel():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    kernel_symbol = "barrier_option_price"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == kernel_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/barrieroption.py"
        and item.symbol == kernel_symbol
        and item.matches_authority
    ]


def test_current_repository_retires_european_swaption_wrapper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == "price_swaption_black76"
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/swaption.py"
        and item.symbol == "price_swaption_black76"
    ]


def test_current_repository_retires_european_swaption_tree_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    retired_symbols = {"price_swaption_tree", "build_swaption_tree_spec"}

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in retired_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/swaption.py"
        and item.symbol in retired_symbols
    ]


def test_current_repository_retires_bermudan_swaption_lower_bound_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_bermudan_swaption_black76_lower_bound"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/bermudanswaption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_bermudan_swaption_tree_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_bermudan_swaption_tree"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/bermudanswaption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_european_swaption_monte_carlo_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    retired_symbols = {
        "price_swaption_monte_carlo",
        "resolve_swaption_monte_carlo_problem",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in retired_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/swaption.py"
        and item.symbol in retired_symbols
    ]


def test_current_repository_retires_analytical_digital_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_digital_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/digitaloption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_chooser_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_chooser_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/chooseroption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_compound_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_compound_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/compoundoption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_lookback_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_fixed_lookback_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/lookbackoption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_variance_swap_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbols = {
        "price_equity_variance_swap_analytical",
        "equity_variance_swap_outputs_analytical",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in helper_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/varianceswap.py"
        and item.symbol in helper_symbols
    ]
