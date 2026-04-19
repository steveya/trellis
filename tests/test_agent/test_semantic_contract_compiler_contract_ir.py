from __future__ import annotations

from trellis.agent.contract_ir import ContractIR
from trellis.agent.platform_requests import compile_build_request
from trellis.agent.semantic_contract_compiler import compile_semantic_contract
from trellis.agent.semantic_contracts import (
    make_american_option_contract,
    make_rate_style_swaption_contract,
    make_vanilla_option_contract,
)


class TestSemanticCompilerContractIR:
    def test_vanilla_semantic_blueprint_attaches_contract_ir(self):
        contract = make_vanilla_option_contract(
            description="European call on AAPL strike 150 expiring 2025-11-15",
            underliers=("AAPL",),
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
        )

        blueprint = compile_semantic_contract(contract)

        assert isinstance(blueprint.contract_ir, ContractIR)
        assert blueprint.contract_ir.underlying.spec.name == "AAPL"

    def test_swaption_semantic_blueprint_attaches_contract_ir(self):
        contract = make_rate_style_swaption_contract(
            description="European payer swaption on 5Y USD IRS strike 5% expiring 2025-11-15",
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
        )

        blueprint = compile_semantic_contract(contract)

        assert isinstance(blueprint.contract_ir, ContractIR)
        assert blueprint.contract_ir.underlying.spec.name == "USD-IRS-5Y"

    def test_out_of_family_semantic_contract_attaches_none(self):
        contract = make_american_option_contract(
            description="American put on AAPL strike 150 expiring 2025-11-15",
            underliers=("AAPL",),
            observation_schedule=("2025-11-15",),
            preferred_method="rate_tree",
            exercise_style="american",
        )

        blueprint = compile_semantic_contract(contract)

        assert blueprint.contract_ir is None

    def test_contract_ir_is_route_independent_across_preferred_methods(self):
        contract = make_vanilla_option_contract(
            description="European call on AAPL strike 150 expiring 2025-11-15",
            underliers=("AAPL",),
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
        )

        analytical = compile_semantic_contract(contract, preferred_method="analytical")
        monte_carlo = compile_semantic_contract(contract, preferred_method="monte_carlo")

        assert analytical.contract_ir == monte_carlo.contract_ir

    def test_compile_build_request_metadata_surfaces_contract_ir(self):
        compiled = compile_build_request(
            "European call on AAPL strike 150 expiring 2025-11-15",
            instrument_type="european_option",
        )

        summary = compiled.request.metadata["semantic_blueprint"]
        assert summary["contract_ir"] is not None
