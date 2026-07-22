.. _fpml-support-matrix:

FpML Support Matrix
===================

**Support-contract version:** 1.0.0

**Effective date:** 2026-07-21

This matrix is the authoritative public support claim for the bounded FpML
import surface. FpML is an external input format. It is not Trellis' task
language, semantic IR, model selector, or pricing engine.

In prose, this documentation uses **FpML 5.13** for the specification version.
XML and request examples preserve the serialized value ``fpmlVersion="5-13"``
and ``source_version="5-13"``. ``dataDocument`` is always written with its
schema capitalization. ``cap/floor`` names the external product cohort;
``period_rate_option_strip`` and ``PeriodRateOptionStripLeg`` are the canonical
Trellis semantic family and leg type.

Support Labels
--------------

``Inspected``
   Secure parsing admitted the document profile and extracted body-free
   document, trade, party, and product identity. It does not mean the product
   can be normalized or priced.

``Normalized``
   The complete admitted economics mapped to an existing Trellis
   ``StaticLegContractIR`` or ``ContractIR`` and received a canonical economic
   identity.

``Executable``
   The normalized contract selected an existing structural declaration,
   validation bundle, callable binding, and deterministic pricing path.

``Conformance-proven``
   A checked task compares independently specified native terms with the FpML
   path across economic projection, identity, structural selection, market
   binding, price, and envelope invariance.

``Blocked``
   The request stops before pricing with a stable blocker and, where caller
   input can resolve the gap, a clarification payload. A blocked document has
   no price and does not invoke an agent recovery path.

Document Profiles
-----------------

.. list-table:: Version, view, and root support
   :header-rows: 1
   :widths: 14 16 20 22 28

   * - Version
     - View
     - Message root
     - Outcome
     - Evidence
   * - FpML 5.13
     - ``confirmation``
     - ``dataDocument``
     - Inspected for one inline direct trade; product support is listed below.
     - `inspection tests`_; `swap fixture`_
   * - FpML 5.13
     - ``recordkeeping``
     - any
     - Blocked with ``external_import:fpml_unsupported_view``.
     - `recordkeeping fixture`_; `conformance tests`_
   * - FpML 5.12
     - ``confirmation``
     - any
     - Blocked with ``external_import:fpml_unsupported_version``.
     - `5.12 fixture`_; `conformance tests`_
   * - FpML 5.13
     - ``confirmation``
     - any root other than ``dataDocument``
     - Blocked with ``external_import:fpml_unsupported_message_root``.
     - `inspection tests`_
   * - any
     - any
     - missing or non-FpML namespace
     - Blocked as a missing namespace or unsupported namespace.
     - `inspection tests`_

The admitted profile uses the
``http://www.fpml.org/FpML-5/confirmation`` namespace. Trellis does not perform
complete XSD validation. Declared view/version and document namespace/version
must agree exactly.

Product Cohorts
---------------

.. list-table:: Product support inside the admitted profile
   :header-rows: 1
   :widths: 22 24 23 31

   * - FpML product shape
     - Normalized semantic shape
     - Highest support level
     - Evidence
   * - Regular single-currency constant-notional fixed-float ``swap``
     - Signed fixed and floating legs in ``StaticLegContractIR``
     - Executable; one signed valuation-party example is conformance-proven
     - `swap normalization tests`_; `swap fixture`_; `conformance tasks`_
   * - Physically settled European payer/receiver ``swaption`` over the
       admitted swap cohort
     - ``ContractIR`` with the complete swap in ``underlying_contract``
     - Executable; the long payer example is conformance-proven
     - `swaption normalization tests`_; `swaption fixture`_; `conformance tasks`_
   * - Regular single-currency constant-strike ``capFloor`` representing one
       cap or one floor strip
     - Signed ``PeriodRateOptionStripLeg`` in ``StaticLegContractIR``
     - Executable; the buyer cap example is conformance-proven
     - `cap-floor normalization tests`_; `cap-floor fixture`_; `conformance tasks`_
   * - Another schema-defined direct product element
     - none
     - Inspected only; normalization blocks with
       ``external_import:fpml_product_normalizer_unavailable``.
     - `imported-request tests`_
   * - ``genericProduct``, ``nonSchemaProduct``, or ``standardProduct``
     - none
     - Blocked during inspection because the wrapper is not complete confirmed
       economics.
     - `inspection tests`_; `generic-product fixture`_;
       `non-schema fixture`_

The exact convention envelope for the three executable cohorts is documented
in :doc:`fpml_import`. In particular, successful inspection of a product name
does not admit its economics. Unsupported product and convention shapes never
fall back to a nearby Trellis family.

Document And Lifecycle Scope
----------------------------

.. list-table:: Document-state support
   :header-rows: 1
   :widths: 25 35 40

   * - Shape
     - Outcome
     - Evidence
   * - One direct trade describing current confirmed economics
     - Admitted when its product and conventions match an executable row.
     - `inspection tests`_; `conformance tasks`_
   * - Missing trade, multiple direct trades, or multiple direct products
     - Blocked as a contract gap or ambiguity.
     - `inspection tests`_
   * - Amendment, increase, cancellation/correction, novation, originating
       event, or termination content
     - Blocked with ``external_import:fpml_unsupported_lifecycle_content``.
     - `lifecycle fixture`_; `inspection tests`_
   * - Package, book, netting, or multi-trade pricing semantics
     - Blocked; no package aggregation or netting is inferred.
     - `inspection tests`_; :ref:`fpml-import-fail-closed`
   * - Recordkeeping-derived current state
     - Blocked because the view and lifecycle-state projection are not yet
       admitted.
     - `recordkeeping fixture`_; `conformance tests`_

Request And Security Contract
-----------------------------

- Requests must supply inline UTF-8 XML bytes. ``source_reference`` is
  provenance only; Trellis never dereferences it and performs no network or
  external-entity access. See `imported-request tests`_.
- DTD and entity declarations are rejected. Default limits are 2,000,000
  bytes, 20,000 elements, and nesting depth 128. See `security tests`_.
- Pricing requires exactly one ``TradeParty`` with
  ``role="valuation_party"`` and a deterministic valuation date. Missing
  caller-owned terms produce clarification blockers. See
  `imported-request tests`_ and `conformance tasks`_.
- Valuation intent, requested outputs, model preference, market snapshot,
  validation policy, execution policy, and expected outcome remain
  ``PlatformRequest`` fields outside the FpML body. FpML labels cannot select
  a route.
- Historical settled premiums are reported as provenance and excluded from
  canonical identity and structural selection. Unsettled premiums block
  because the current payoff path does not model that cashflow. See
  `swaption normalization tests`_ and `cap-floor normalization tests`_.
- Raw XML and parser nodes are not persisted in import summaries or task-run
  evidence. Reports retain digests, body-free identity, provenance paths,
  normalized summaries, and blockers. See `imported-request tests`_ and
  `conformance tests`_.

Change Control
--------------

Changing an ``Inspected``, ``Normalized``, ``Executable``, or
``Conformance-proven`` claim requires all of the following in one reviewed
slice:

1. secure positive and negative fixtures
2. deterministic importer and normalization tests
3. an existing semantic home and executable lowering path
4. paired native/FpML identity, selection, price, and blocker evidence
5. updates to this matrix, :doc:`fpml_import`, user/quant documentation, and
   ``LIMITATIONS.md``
6. the release validation gate

An import extension must not add a product-specific pricing helper, generated
adapter authority, hand-authored cookbook route, or FpML-label dispatch rule.

.. _inspection tests: https://github.com/steveya/trellis/blob/main/tests/test_io/test_fpml_import.py
.. _security tests: https://github.com/steveya/trellis/blob/main/tests/test_io/test_fpml_import.py
.. _swap normalization tests: https://github.com/steveya/trellis/blob/main/tests/test_io/test_fpml_fixed_float_swap.py
.. _swaption normalization tests: https://github.com/steveya/trellis/blob/main/tests/test_io/test_fpml_european_swaption.py
.. _cap-floor normalization tests: https://github.com/steveya/trellis/blob/main/tests/test_io/test_fpml_cap_floor.py
.. _imported-request tests: https://github.com/steveya/trellis/blob/main/tests/test_agent/test_imported_document_requests.py
.. _conformance tests: https://github.com/steveya/trellis/blob/main/tests/test_agent/test_fpml_conformance.py
.. _conformance tasks: https://github.com/steveya/trellis/blob/main/TASKS_FPML_CONFORMANCE.yaml
.. _swap fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_13_fixed_float_swap.xml
.. _swaption fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_13_european_swaption.xml
.. _cap-floor fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_13_cap_floor.xml
.. _generic-product fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_13_generic_product.xml
.. _non-schema fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_13_non_schema_product.xml
.. _lifecycle fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_13_lifecycle.xml
.. _recordkeeping fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/recordkeeping_5_13_unsupported.xml
.. _5.12 fixture: https://github.com/steveya/trellis/blob/main/tests/test_io/fixtures/fpml/confirmation_5_12_unsupported.xml
