# Barrier Kernel Pack Proof of Concept

This note is the mathematical proof-of-concept for `QUA-328`.

The immediate goal is narrow: show that the T09 barrier route can be assembled
from a small reusable kernel pack rather than implemented as one monolithic
formula block. The broader architectural goal is to describe that assembly in a
more formal language suitable for Trellis: contracts as syntax, valuation as
semantics, and analytical decomposition as a semantics-preserving rewrite under
explicit assumptions.

This note still does **not** introduce a generic barrier product family. It is
intentionally a route-scoped proof-of-concept.

The first pass only needs the T09 regime:

- down-and-out call
- continuous monitoring
- zero rebate
- `S_0 = 100`, `K = 100`, `B = 90`, `r = 0.05`, `sigma = 0.20`, `T = 1`

That is exactly the case exercised by `tests/test_tasks/test_t09_barrier.py`.

## 1. Foundational language

This section introduces the minimal formal language used in the rest of the
note. The intent is Lean-inspired in discipline, but not dependent on Lean or
any theorem-prover integration.

### 1.1 Syntax, semantics, and evaluators

A Trellis route should be understood in three layers.

1. **Contract syntax**: the structured expression that describes the claim.
2. **Valuation semantics**: the mathematical value of that claim under a stated
   model and exercise regime.
3. **Evaluator**: a concrete exact or approximate method used to compute that
   value.

Under this view, a down-and-out call is not "the Reiner-Rubinstein formula."
It is first a contract expression. The Reiner-Rubinstein-style closed form is
an evaluator or valuation lemma for that contract under a specific model class
and side conditions.

### 1.2 Kernels

A **kernel** is a primitive valuation component with a fixed semantic meaning
and a stated validity envelope.

For this proof-of-concept, the intended kernels are:

- `vanilla_call_raw`: the Black-Scholes value of a European call payoff
- `barrier_image_raw`: the image-term contribution for the continuous
  down-barrier regime addressed here
- `rebate_raw`: a rebate contribution when the route requires one
- `barrier_regime_selector_raw`: a non-smooth dispatcher over valid analytical
  branches

A kernel is therefore not just a reusable formula fragment. It is a reusable
priced claim-component or valuation component with explicit assumptions.

### 1.3 Decomposition as rewrite

A **decomposition** in Trellis is a semantics-preserving rewrite from one
contract valuation expression into a composition of simpler valuation
components, under stated assumptions.

For this note, the central rewrite is:

$$
\text{DownAndOutCall} \leadsto \text{VanillaCall} - \text{BarrierImage}
$$

in the continuously monitored, zero-rebate, `K > B`, Black-Scholes regime.

This should be read as more than an algebraic factorization of a final formula.
It is a route-local valuation decomposition into reusable analytical kernels.

### 1.4 Side conditions and validity envelopes

Every analytical kernel or rewrite must carry side conditions. In this note,
those side conditions include model assumptions, monitoring conventions,
parameter inequalities, and regime restrictions.

A formula is therefore not merely "available." It is available only when its
side conditions are satisfied, or when the public adapter / selector has routed
control to a branch whose preconditions are known to hold.

### 1.5 Boundary between differentiable core and public adapter

Trellis should separate:

- the **smooth analytical core**, where a kernel is evaluated on its open domain
- the **public adapter / selector**, where non-smooth dispatch decisions,
  boundary handling, and fallbacks occur

This note uses that separation explicitly. The raw kernels live on the smooth
interior; the public route surface handles barrier-breach checks, regime
selection, and any out-of-domain behavior.

### 1.6 Pricing algebra and basis claims

For a terminal-state payoff $g(X)$, define the pricing functional

$$
\Pi[g] = e^{-rT}\mathbb E^Q[g(X)].
$$

If the payoff can be written as a finite linear combination

$$
g(x) = \sum_{i=1}^n a_i \phi_i(x),
$$

then by linearity of expectation,

$$
\Pi[g] = \sum_{i=1}^n a_i \Pi[\phi_i].
$$

For Black-Scholes style valuation, useful basis claims include:

- cash-or-nothing digital call: $\mathbf{1}_{\{x > K\}}$
- cash-or-nothing digital put: $\mathbf{1}_{\{x < K\}}$
- asset-or-nothing call: $x\,\mathbf{1}_{\{x > K\}}$
- asset-or-nothing put: $x\,\mathbf{1}_{\{x < K\}}$

These give exact finite decompositions for the familiar vanilla payoffs:

$$
(x-K)^+ = x\,\mathbf{1}_{\{x > K\}} - K\,\mathbf{1}_{\{x > K\}},
$$

$$
(K-x)^+ = K\,\mathbf{1}_{\{x < K\}} - x\,\mathbf{1}_{\{x < K\}}.
$$

This is the precise sense in which Trellis can assemble a pricing formula from
reusable analytical kernels: the final contract value is a linear combination
of basis claims with known prices.

The barrier POC in this note uses the same idea at the route level:
`vanilla_call_raw` and `barrier_image_raw` are the local basis pieces for the
T09 branch, and the public route assembles the final price from them.

## 2. Contract meaning for the T09 route

The T09 route is a continuously monitored down-and-out European call with zero
rebate. At maturity, the claim pays

$$
(S_T - K)^+ \mathbf{1}_{\{\tau_B > T\}},
$$

where

$$
\tau_B = \inf\{ t \ge 0 : S_t \le B \}
$$

is the first hitting time of the down barrier.

For this proof-of-concept, the contract is valued under the risk-neutral
Black-Scholes model and restricted to the branch

$$
S_0 > B, \qquad K > B, \qquad \text{rebate} = 0.
$$

This branch is exactly the one used by T09.

## 3. Model, notation, and validity envelope

Under risk-neutral Black-Scholes dynamics,

$$
dS_t = r S_t \, dt + \sigma S_t \, dW_t,
$$

with no dividend or carry term in this first pass.

Throughout the kernel formulas below, `S` denotes current spot. In the T09 test
case, `S = S_0 = 100`.

The validity envelope for the raw closed-form branch in this note is:

- Black-Scholes lognormal diffusion
- constant `r` and `sigma`
- European payoff
- continuous barrier monitoring
- down barrier with `0 < B < S`
- zero rebate
- `K > B`
- `T > 0`, `sigma > 0`

No claim is made here about correctness outside this envelope. Boundary cases,
regime changes, and unsupported features belong to the selector, adapter, or a
separate fallback engine.

Define the usual Black-Scholes terms:

$$
d_1 = \frac{\log(S/K) + (r + \tfrac12 \sigma^2) T}{\sigma \sqrt{T}},
\qquad
d_2 = d_1 - \sigma \sqrt{T}.
$$

Also define the barrier-image parameters:

$$
\lambda = \frac{r + \tfrac12 \sigma^2}{\sigma^2},
\qquad
y = \frac{\log(B^2/(SK))}{\sigma \sqrt{T}} + \lambda \sigma \sqrt{T},
\qquad
y_2 = y - \sigma \sqrt{T}.
$$

Let $\Phi(\cdot)$ denote the standard normal CDF.

## 4. Kernel semantics

This section states the semantic contract of each raw kernel used in the
proof-of-concept.

### 4.1 Vanilla call kernel

The vanilla call kernel is

$$
C_{\mathrm{BS}}(S, K; r, \sigma, T)
=
S \Phi(d_1) - K e^{-rT} \Phi(d_2).
$$

**Semantic meaning.** This kernel returns the Black-Scholes value of the
European payoff

$$
(S_T - K)^+.
$$

**Role in assembly.** This is the reusable vanilla component from which the
barrier route is built.

### 4.2 Image-term kernel

For the continuously monitored down-barrier call in the `K > B`, zero-rebate
regime, define the image kernel by

$$
I(S, K, B; r, \sigma, T)
=
S \left(\frac{B}{S}\right)^{2 \lambda} \Phi(y)
- K e^{-rT} \left(\frac{B}{S}\right)^{2 \lambda - 2} \Phi(y_2).
$$

**Semantic meaning.** In this regime, the image kernel equals the down-and-in
call value:

$$
C_{\mathrm{DI}} = I.
$$

Equivalently, it is the image / reflection contribution removed from the
vanilla call in order to obtain the down-and-out value.

**Role in assembly.** This is the second reusable primitive for the T09 route.

### 4.3 Rebate kernel

No rebate kernel is used in the current branch because the T09 proof-of-concept
assumes zero rebate.

The point of naming `rebate_raw` anyway is architectural: a rebate contribution
should be treated as its own semantic component rather than fused into the main
barrier formula when future routes require it.

### 4.4 Regime selector

The regime selector is not part of the smooth analytical kernel itself.

**Semantic meaning.** It dispatches to a valid analytical branch only when the
required side conditions are satisfied.

**Role in assembly.** It handles branch distinctions such as `K > B` versus
`K <= B`, along with other route-surface checks such as whether the barrier has
already been breached.

This selector is intentionally non-smooth and belongs outside the
differentiable core.

## 5. Route assembly for T09

In the T09 regime, the down-and-out call value is assembled as

$$
C_{\mathrm{DO}} = C_{\mathrm{BS}} - I.
$$

Equivalently,

$$
\texttt{DownAndOutCall(T09 branch)}
\leadsto
\texttt{vanilla\_call\_raw} - \texttt{barrier\_image\_raw}.
$$

This is the exact meaning of "assembled" in `QUA-328`.

The route is therefore not modeled here as a monolithic bespoke formula block.
It is modeled as a composition of smaller analytical kernels whose semantic
roles are explicit.

At the level of route construction, the assembly is:

1. verify that control is in the T09 branch
2. compute the vanilla call kernel
3. compute the image-term kernel
4. subtract the image term from the vanilla call
5. apply public-surface boundary handling only outside the raw kernel layer

This is a route-local example of the broader Trellis design principle that
analytical support should be expressed as compositional assembly over reusable
valuation components.

## 6. Why this decomposition is mathematically meaningful

The intended reading of

$$
C_{\mathrm{DO}} = C_{\mathrm{BS}} - I
$$

is not merely that the final formula can be split into two algebraic terms.
The stronger claim is that, in the stated model regime:

- the vanilla kernel has a clear claim-semantic meaning
- the image kernel has a clear claim-semantic meaning
- the route value is assembled by a semantics-preserving rewrite into those
  pieces

This distinction matters for Trellis. Formula factorization by itself is not
sufficient justification for a reusable kernel. Reuse becomes meaningful when a
term has a stable semantic role, a clear validity envelope, and route-level
value beyond a single implementation.

## 7. Why this is autograd-friendly

On the open interior of the T09 branch, the raw kernel composition is
differentiable because it is built from:

- `log`
- `exp`
- powers
- the normal CDF
- simple arithmetic composition

The non-smooth points are the hard dispatch boundaries:

- whether the barrier has already been breached
- whether `K > B` or `K <= B`
- any unsupported or fallback-triggering domain boundary

Those switches should stay in the thin public adapter or the regime selector,
not inside the differentiable analytical core.

For the first pass, that is acceptable because the T09 consumer lives in the
smooth `S > B`, `K > B` regime and is already cross-validated against PDE,
Monte Carlo, and QuantLib.

## 8. What is intentionally left out

This proof-of-concept does not attempt to define a generic barrier algebra or a
full shared barrier support package.

It does **not** yet include:

- up-and-out / up-and-in support as a first-class shared layer
- double barriers
- discrete monitoring adjustments
- boundary-point differentiability at `S = B`
- rebate formulas beyond the zero-rebate T09 case
- a universal barrier abstraction independent of model assumptions

Those are follow-on candidates only if later routes reuse the same semantic
components or rewrite patterns.

## 9. Kernel-pack view

The intended kernel pack for the first phase is:

- `vanilla_call_raw`
- `barrier_image_raw`
- `barrier_regime_selector_raw`
- `rebate_raw` when needed

The route-level assembly then becomes a small composition over those kernels
rather than a bespoke product implementation.

In Trellis terms, this is a small trusted analytical pack for one route family:

- a primitive vanilla valuation kernel
- a primitive barrier image valuation kernel
- a selector that discharges branch side conditions
- an optional additive rebate component

That is the minimum mathematical structure needed to justify promoting barrier
work from route-local code into a reusable analytical kernel pack later.

## 10. Relation to the broader Trellis direction

This note is intentionally small, but it fits a broader architectural pattern.

A Trellis analytical route should ideally be described by:

1. a contract expression
2. a stated model and validity envelope
3. a set of named rewrites or decomposition lemmas
4. a kernel assembly over reusable valuation components
5. a thin route interpreter or adapter at the public boundary

This barrier proof-of-concept is therefore not just about one formula. It is a
worked example of how analytical support can be described in a more formal,
compositional vocabulary without requiring theorem-prover infrastructure.

## 11. Relation to the Linear work items

This note supports the following issue sequence:

- `QUA-289` analytical support as a template-plus-delta system
- `QUA-291` reusable analytical kernels
- `QUA-292` thin route interpreters over shared support
- `QUA-293` builder guidance for analytical assembly
- `QUA-328` the concrete barrier consumer that proves whether the kernel pack
  is worth extracting

If a second route later uses the same barrier image term, selector logic, or
related rewrite structure, that is the point where a shared barrier support
layer becomes justified.
