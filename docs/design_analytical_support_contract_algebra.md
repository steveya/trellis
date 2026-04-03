---
title: "Analytical Support via Contract Algebra and Compositional Decomposition"
subtitle: "Formal design document and mathematical theory draft"
author: "OpenAI"
date: "2026-04-02"
lang: en-US
toc: true
toc-depth: 3
geometry: margin=1in
fontsize: 11pt
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: blue
header-includes:
  - |
    \usepackage{amsmath,amssymb,amsthm,mathtools}
  - |
    \usepackage{booktabs,longtable,array}
  - |
    \usepackage{graphicx}
  - |
    \usepackage{fancyhdr}
  - |
    \pagestyle{fancy}
  - |
    \fancyhf{}
  - |
    \fancyhead[L]{Analytical Support via Contract Algebra and Compositional Decomposition}
  - |
    \fancyhead[R]{\thepage}
  - |
    \setlength{\headheight}{14pt}
---

Prepared as a consolidation of the available analytical-support materials, centered on the barrier proof-of-concept for `QUA-328` and on the file-map you supplied for the broader Trellis analytical-support direction.

Current implementation note, 2026-04-03: the contract-algebra boundary is no
longer only an analytical/barrier design story. The same typed semantic and
helper-backed discipline now governs ranked-observation baskets, single-name
CDS, and nth-to-default basket-credit lowering, with the validation contract
loop consuming the resulting family IR and exact helper bindings.

![Analytical support architecture](/mnt/data/analytical_support_architecture.png)

## Scope note

This draft is intentionally honest about source coverage.

The accessible primary source for the formal mathematical material was the barrier proof-of-concept, which explicitly frames the design in terms of **contracts as syntax**, **valuation as semantics**, and **analytical decomposition as a semantics-preserving rewrite**. The exact current-maintained documents named in your file map - `basis_claim_patterns.md`, `differentiable_pricing.rst`, `analytical_route_cookbook.rst`, and `contract_algebra.rst` - were not directly available in the material I could inspect while preparing this draft. Accordingly:

- everything specific to the T09 barrier route is presented as a direct consolidation of accessible source material;
- everything broader than the barrier route is presented as a **formal synthesis draft**, designed to be consistent with the direction named in your file map but not pretending to quote inaccessible text;
- the historical `CompositeSemanticContract` / DAG-style composition note is treated as background only, not as current authority.

The resulting document is therefore best read as a **working authority draft**: formal enough to guide implementation and review, but still suitable for refinement once the additional maintained documents are folded in.

# 1. Executive summary

The central design claim is that analytical support should not be organized as a bag of route-specific formulas. It should be organized as a small algebra over contracts, kernels, and sound rewrites.

The governing distinction is:

1. **contract syntax** - the claim that is being priced;
2. **valuation semantics** - the mathematical value of that claim under a stated model and exercise regime;
3. **evaluator** - the concrete exact or approximate procedure used to compute that semantic value.

Under this view, a closed form such as a Reiner-Rubinstein barrier expression is not the contract. It is an evaluator, or more precisely a valuation lemma valid under an explicit model class and side conditions.

The main architectural consequence is that a reusable analytical component is justified only when it has all three of the following:

- a stable **semantic role**;
- a clearly stated **validity envelope**;
- route-level **reuse value** beyond a single implementation.

This produces the following design discipline.

- Analytical support is expressed as **kernel assembly** rather than monolithic formula blocks.
- Decomposition is expressed as a **semantics-preserving rewrite** rather than as an incidental algebraic factorization.
- The **smooth differentiable core** is separated from the **public adapter / selector**, which is allowed to contain non-smooth dispatch, boundary handling, and fallbacks.
- Shared analytical support is promoted only when route-local mathematics stabilizes into semantically named reusable components.

The worked example in this document is the T09 continuously monitored down-and-out call with zero rebate in the branch

$$
S_0 > B, \qquad K > B,
$$

for which the route value is assembled as

$$
C_{\mathrm{DO}} = C_{\mathrm{BS}} - I,
$$

or equivalently

$$
\texttt{DownAndOutCall(T09 branch)}
\leadsto
\texttt{vanilla\_call\_raw} - \texttt{barrier\_image\_raw}.
$$

The architectural meaning of this equation is stronger than “the formula happens to split into two terms.” It means that the down-and-out route can be discharged by a sound rewrite into reusable valuation components with explicit claim-semantic roles.

# 2. Source basis and authority boundary

## 2.1 Directly consolidated material

This draft directly consolidates the accessible barrier proof-of-concept material associated with `QUA-328`, including the following ideas.

- Contracts are treated as syntax, valuation as semantics, and analytical decomposition as semantics-preserving rewrite.
- A kernel is a primitive valuation component with fixed semantic meaning and a stated validity envelope.
- The T09 barrier branch is valued under Black-Scholes on the explicit branch
  $$
  S_0 > B, \qquad K > B, \qquad \text{rebate}=0.
  $$
- The down-and-out route is assembled from a vanilla kernel and a barrier image kernel.
- Non-smooth regime checks are placed outside the differentiable raw core.
- A generic barrier algebra is intentionally **not** claimed at this stage.

## 2.2 Synthesis beyond the barrier route

Your file map identifies four additional maintained design loci.

- `basis_claim_patterns.md`
- `differentiable_pricing.rst`
- `analytical_route_cookbook.rst`
- `contract_algebra.rst`

Because those documents were not directly available for inspection, this draft does not claim to reproduce their exact wording. Instead, it uses them as named anchors for a broader formal synthesis with four goals.

- to make the contract-algebra vocabulary explicit;
- to make the rewrite theory precise;
- to define the smooth-core / public-adapter boundary as a general rule;
- to articulate a route cookbook that can host future route families without collapsing into formula sprawl.

## 2.3 Historical material

The older composition / DAG writeup is treated as historical background, not as normative authority. Where this draft uses algebraic language for composition, it does so in a way that is compatible with the current barrier proof-of-concept rather than by reviving a historical abstraction for its own sake.

# 3. Foundational language

## 3.1 Contracts as syntax

A contract is first a structured expression describing a claim. It is not yet a number, and it is not yet a chosen numerical method.

We write a contract grammar schematically as

$$
C ::= 0
\mid a \cdot C
\mid C_1 + C_2
\mid \operatorname{Payoff}(\psi)
\mid \operatorname{VanillaCall}(K,T)
\mid \operatorname{DownAndOutCall}(K,B,T)
\mid \cdots
$$

where the final list of primitives is route-family dependent.

The point of the syntax layer is not to freeze the final contract language. The point is to preserve the distinction between:

- the claim that exists mathematically;
- the representation used to talk about that claim;
- the evaluator later chosen to compute its value.

## 3.2 Valuation as semantics

Fix a model regime $M$, a market state $x$, and an admissible contract $C$. The valuation semantics is a map

$$
\left[\!\left[ C \right]\!\right]_M(x) \in \mathbb{R}
$$

giving the mathematical value of the claim under the stated model and exercise regime.

This notation is deliberately semantic. It asserts that the number belongs to the model-and-claim pair, not to any particular implementation.

For example, under risk-neutral Black-Scholes on the admissible T09 branch,

$$
\left[\!\left[ \operatorname{DownAndOutCall}(K,B,T) \right]\!\right]_{\mathrm{BS}}(x)
$$

means the value of the continuously monitored zero-rebate down-and-out call under Black-Scholes, not “whatever some route function currently returns.”

## 3.3 Evaluators

An evaluator is a concrete method that computes or approximates a semantic value.

Examples include:

- a route-local closed form;
- a kernel assembly obtained from sound rewrite;
- a PDE engine;
- a Monte Carlo engine;
- a tree or lattice method;
- a fallback numerical adapter.

The right correctness judgment is not “this is a good formula.” It is:

> **Does this evaluator agree with the valuation semantics on its stated domain?**

Formally, an evaluator $E$ is sound for a contract $C$ on domain $D$ if

$$
E_M(C,x)=\left[\!\left[ C \right]\!\right]_M(x)
\qquad \text{for all } x \in D.
$$

## 3.4 Kernels

A kernel is a primitive valuation component with fixed semantic meaning and a stated validity envelope.

That definition excludes two bad extremes.

A kernel is **not** merely an arbitrary formula fragment.

A kernel is also **not** required to be a fully generic product family.

Instead, a kernel is the right granularity when the component has:

- stable semantic interpretation,
- clear side conditions,
- independent route value,
- acceptable testing surface.

For the barrier proof-of-concept, the intended kernels are:

- `vanilla_call_raw`,
- `barrier_image_raw`,
- `rebate_raw`,
- `barrier_regime_selector_raw`.

## 3.5 Side conditions and validity envelopes

Every closed form, kernel, and rewrite must be paired with a validity envelope.

A validity envelope may constrain any of the following.

- model class,
- monitoring convention,
- payoff type,
- barrier direction,
- rebate status,
- parameter inequalities,
- smooth-interior conditions,
- required branch selection.

The barrier proof-of-concept makes this explicit. The T09 raw branch assumes:

- Black-Scholes lognormal diffusion,
- constant $r$ and $\sigma$,
- European payoff,
- continuous barrier monitoring,
- down barrier,
- zero rebate,
- $K > B$,
- $T > 0$ and $\sigma > 0$.

A formula is therefore not merely “available.” It is available **only** when its envelope holds, or when control has already been routed into a branch known to satisfy the required preconditions.

# 4. Contract algebra and pricing algebra

## 4.1 Contract algebra

A minimal contract algebra is generated by additive and scalar structure together with semantically named basis claims.

We define the algebraic constructors

$$
0, \qquad a \cdot C, \qquad C_1 + C_2
$$

with the intended semantic laws

$$
\left[\!\left[ 0 \right]\!\right]_M = 0,
$$

$$
\left[\!\left[ a \cdot C \right]\!\right]_M = a \, \left[\!\left[ C \right]\!\right]_M,
$$

$$
\left[\!\left[ C_1 + C_2 \right]\!\right]_M = \left[\!\left[ C_1 \right]\!\right]_M + \left[\!\left[ C_2 \right]\!\right]_M,
$$

whenever the right-hand side is well-defined under the same regime.

This algebraic layer is the right place to express portfolio-like claim composition, parity relations, and route decompositions.

## 4.2 Basis claims

A **basis claim** is a syntactic primitive whose semantic interpretation is stable enough to serve as a reusable analytical building block.

In the barrier proof-of-concept, two such basis claims are already visible.

- the European vanilla call payoff,
- the down-and-in image contribution on the stated branch.

More generally, a route family earns a reusable basis-claim vocabulary only when those claims can be named without secretly smuggling in the final route formula.

This matters because “basis claim” is stronger than “term that appears in one derivation.” A basis claim should support at least one of the following.

- semantic reuse across multiple routes,
- stable testing in isolation,
- meaningful AD exposure,
- intelligible public documentation.

## 4.3 Pricing algebra

Given a contract algebra and a library of sound evaluators, pricing algebra is the discipline of expressing valuation by compositions of semantically named kernel evaluations.

Concretely, if

$$
A = K_1 \oplus K_2 \oplus \cdots \oplus K_n
$$

is a kernel assembly expression, then its semantics is induced compositionally from the semantics of the participating kernels and assembly operators.

The important restriction is that the assembly operators must be semantically meaningful. Addition, subtraction, scalar multiplication, and branch selection are meaningful. Ad hoc symbolic surgery is not.

## 4.4 Why algebraic structure matters

The algebraic viewpoint brings four immediate benefits.

First, it localizes correctness.

Second, it allows sound reuse.

Third, it exposes the true differentiability boundary.

Fourth, it prevents route implementations from confusing “one derivation happened to work” with “we have extracted a reusable analytical component.”

# 5. Rewrite theory

## 5.1 Rewrite judgment

A rewrite is written schematically as

$$
\Gamma \vdash C \rightsquigarrow A,
$$

where:

- $C$ is a contract valuation target,
- $A$ is a kernel assembly expression,
- $\Gamma$ is the set of side conditions under which the rewrite is admissible.

The intended meaning is:

> Under side conditions $\Gamma$, the contract valuation target $C$ may be replaced by the assembly $A$ without changing semantic value.

## 5.2 Semantic soundness

**Proposition 5.1 (soundness of admissible rewrite).**  
If

$$
\Gamma \vdash C \rightsquigarrow A
$$

and the model state $x$ satisfies $\Gamma$, then

$$
\left[\!\left[ C \right]\!\right]_M(x)=\left[\!\left[ A \right]\!\right]_M(x).
$$

**Proof sketch.**  
This is the governing proof obligation attached to the rewrite itself. Either it is discharged by theorem, by parity identity, by model-specific derivation, or by a previously trusted valuation lemma. The rewrite is admitted only once that obligation is met. $\square$

The key point is conceptual: a decomposition is not “allowed because it looks plausible.” It is admitted only as a sound semantic replacement.

## 5.3 Route-local versus shared rewrites

Not every sound rewrite should become a public shared abstraction.

A rewrite is **route-local** when its semantic content is valid but not yet broad enough to justify a reusable shared layer.

A rewrite becomes **shared** only when:

- its semantic components have stable names,
- its validity envelope is well understood,
- at least one additional route plausibly reuses the same structure, or its isolated testing value is already high.

The barrier proof-of-concept is explicit on this point: it does **not** yet claim a generic barrier algebra. It claims only that the T09 route admits a small trusted kernel pack.

## 5.4 Promotion criteria for reusable support

A route-local formula fragment should be promoted into shared analytical support only if all of the following hold.

1. It has a stable semantic role.
2. Its side conditions can be stated explicitly.
3. It is useful across more than one route, or clearly worthwhile as an independently tested primitive.
4. Its interface can be kept small and branch-clean.
5. It does not force non-smooth selector logic into the raw differentiable core.

# 6. Differentiable pricing boundary

## 6.1 Smooth core versus public adapter

A crucial design rule is the separation between:

- the **smooth analytical core**, where kernels are evaluated on their open domain,
- the **public adapter / selector**, where non-smooth dispatch, boundary handling, unsupported-feature checks, and fallbacks occur.

This boundary is not merely an implementation preference. It is part of the mathematical theory.

If branch checks are fused into the raw algebraic core, then one loses a clean statement of differentiability and a clean statement of evaluator soundness on the intended branch interior.

## 6.2 Interior differentiability principle

**Proposition 6.1 (interior differentiability of raw kernel assembly).**  
Let $A$ be a kernel assembly built only from smooth arithmetic composition, exponentials, logarithms, powers, and the normal CDF. Then $A$ is differentiable on any open subset of parameter space that avoids dispatch boundaries and formula singularities.

**Proof sketch.**  
The listed primitives are differentiable on their natural open domains. Finite compositions of differentiable maps remain differentiable. Therefore the only non-smoothness arises from branch switches, breach checks, discrete monitoring approximations, unsupported-feature fallbacks, or explicit max/min-type structure introduced outside the interior branch. $\square$

## 6.3 Engineering corollary

The engineering corollary is simple.

Do not push dispatch logic into `*_raw` kernels.

Instead:

- keep `*_raw` kernels branch-pure,
- keep selectors explicit,
- let the public adapter own branch routing and fallbacks.

# 7. Analytical route cookbook

This section generalizes the implementation pattern named in your file map.

## 7.1 Layer pattern

A disciplined analytical route should be organized as:

$$
\text{resolver} \rightarrow \text{support helpers} \rightarrow \text{raw kernels} \rightarrow \text{public adapter}.
$$

The layers play distinct roles.

| Layer | Primary responsibility | May be non-smooth? | Owns semantic authority? |
|---|---|---:|---:|
| Resolver | choose route family / branch candidate | yes | no |
| Support helpers | compute branch parameters, normalize conventions, enforce envelopes | possibly | no |
| Raw kernels | evaluate semantically named primitives on open domain | no, except intrinsic formula limits | yes |
| Public adapter | dispatch, boundary handling, fallback, surface API | yes | no |

## 7.2 Resolver

The resolver answers questions such as:

- which model regime is active,
- which analytical branch is potentially valid,
- whether exact support exists or fallback is required.

Its output is not the price. Its output is a justified branch candidate plus the information required to attempt raw-kernel evaluation.

## 7.3 Support helpers

Support helpers convert route inputs into the branch-normalized quantities required by raw kernels.

Examples include:

- standardized Black-Scholes quantities,
- transformed barrier coordinates,
- parity-related helper quantities,
- model-normalized inputs for route-local lemmas.

The helpers should remain mathematically transparent. They are not the place to hide branch changes.

## 7.4 Raw kernels

Raw kernels should satisfy the following.

- semantically named,
- branch-pure,
- validity envelope stated,
- independently testable,
- AD-friendly on the open interior.

## 7.5 Public adapter

The public adapter may legitimately own:

- barrier-breach checks,
- unsupported-feature messages,
- fallback to PDE or Monte Carlo,
- boundary-specific behavior,
- user-facing normalization.

It should not secretly change the mathematics of the raw kernels.

# 8. Worked example: the T09 barrier route

## 8.1 Contract and model

The T09 route is the continuously monitored down-and-out European call with zero rebate and payoff

$$
(S_T-K)^+ \mathbf{1}_{\{\tau_B>T\}},
$$

where

$$
\tau_B = \inf\{t \ge 0 : S_t \le B\}
$$

is the first hitting time of the down barrier.

The valuation regime is the risk-neutral Black-Scholes model

$$
dS_t = r S_t \, dt + \sigma S_t \, dW_t,
$$

with the branch restrictions

$$
S_0 > B, \qquad K > B, \qquad \text{rebate}=0.
$$

For this first-pass proof-of-concept, no dividend or carry term is included.

## 8.2 Branch-normalized notation

Define the usual Black-Scholes quantities

$$
d_1 = \frac{\log(S/K) + (r + \tfrac12 \sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}.
$$

Define further

$$
\lambda = \frac{r + \tfrac12 \sigma^2}{\sigma^2},
\qquad
y = \frac{\log(B^2/(SK))}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T},
\qquad
y_2 = y - \sigma\sqrt{T}.
$$

Let $\Phi$ denote the standard normal CDF.

## 8.3 Vanilla kernel

Define the vanilla call kernel

$$
C_{\mathrm{BS}}(S,K;r,\sigma,T)
=
S\Phi(d_1)-Ke^{-rT}\Phi(d_2).
$$

**Semantic meaning.**  
This kernel returns the Black-Scholes value of the European payoff

$$
(S_T-K)^+.
$$

## 8.4 Image-term kernel

Define the image kernel on the admissible T09 branch by

$$
I(S,K,B;r,\sigma,T)
=
S\left(\frac{B}{S}\right)^{2\lambda}\Phi(y)
-
Ke^{-rT}\left(\frac{B}{S}\right)^{2\lambda-2}\Phi(y_2).
$$

**Semantic meaning.**  
On this branch, the image kernel equals the down-and-in call value:

$$
C_{\mathrm{DI}} = I.
$$

Equivalently, it is the image / reflection contribution that must be removed from the vanilla call in order to obtain the down-and-out value.

## 8.5 Rebate kernel

No rebate kernel is used in the current branch because the proof-of-concept is explicitly restricted to zero rebate.

The architectural reason to keep the name `rebate_raw` alive is that future routes may need rebate contributions as independent semantic components rather than as terms fused inside a product-specific master formula.

## 8.6 Regime selector

The regime selector is deliberately outside the smooth raw core.

Its semantic role is to dispatch to a valid analytical branch only when the required side conditions are satisfied. In the T09 family this includes, at minimum:

- barrier not already breached,
- branch distinction $K > B$ versus $K \le B$,
- rejection or fallback outside supported features.

## 8.7 Assembly rewrite

The route value on the T09 branch is assembled as

$$
C_{\mathrm{DO}} = C_{\mathrm{BS}} - I.
$$

Equivalently,

$$
\texttt{DownAndOutCall(T09 branch)}
\rightsquigarrow
\texttt{vanilla\_call\_raw} - \texttt{barrier\_image\_raw}.
$$

This is the central worked example of semantics-preserving rewrite.

## 8.8 Soundness proposition

**Proposition 8.1 (T09 assembly soundness).**  
On the Black-Scholes continuously monitored zero-rebate branch with $S_0 > B$ and $K > B$,

$$
\left[\!\left[ \operatorname{DownAndOutCall}(K,B,T) \right]\!\right]_{\mathrm{BS}}
=
C_{\mathrm{BS}} - I.
$$

**Proof sketch.**  
By in-out parity on the admissible branch,

$$
C_{\mathrm{Vanilla}} = C_{\mathrm{DownOut}} + C_{\mathrm{DownIn}}.
$$

The branch-specific image expression identifies

$$
C_{\mathrm{DownIn}} = I.
$$

Rearranging gives

$$
C_{\mathrm{DownOut}} = C_{\mathrm{Vanilla}} - I = C_{\mathrm{BS}} - I.
$$

Hence the kernel assembly is a sound semantic rewrite of the original route value. $\square$

## 8.9 Why the decomposition is mathematically meaningful

The decomposition is not important merely because the final formula contains two terms.

It is important because the terms have distinct semantic roles.

- The vanilla kernel is the value of a well-understood basis claim.
- The image kernel is the value of a distinct reflected / down-in contribution on the stated branch.
- The route price is produced by a sound semantic subtraction.

This is precisely what makes the kernel pack reusable. Formula factorization alone would not justify extraction.

## 8.10 Differentiability proposition

**Proposition 8.2 (interior differentiability of the T09 raw assembly).**  
On the open branch interior where

$$
S > B, \qquad K > B, \qquad T > 0, \qquad \sigma > 0,
$$

the raw assembly

$$
C_{\mathrm{BS}} - I
$$

is differentiable with respect to its continuous parameters.

**Proof sketch.**  
Both $C_{\mathrm{BS}}$ and $I$ are built from arithmetic operations, exponentials, logarithms, powers, and the normal CDF, each on its natural open domain. Their difference is therefore differentiable on the open branch interior. Non-smoothness enters only through branch-selection and boundary events, which are assigned to the selector / adapter layer rather than to the raw kernel itself. $\square$

# 9. Generalization pattern beyond the barrier route

The barrier example is deliberately narrow, but the design pattern generalizes.

A future analytical route family - for example FX, quanto, Jamshidian-style decomposition, or other closed-form branches named in your file map - should be admitted into the same framework only when it can supply the following.

1. A contract expression.
2. A model regime and validity envelope.
3. A set of semantically named kernels.
4. One or more sound rewrites from route value to kernel assembly.
5. A clear smooth-core / public-adapter boundary.

The framework does **not** require that all route families share the same kernels. It requires only that every shared kernel be semantically stable and envelope-explicit.

# 10. Non-goals and deliberate exclusions

This draft does not claim any of the following.

- a universal barrier algebra,
- generic correctness outside the stated branch envelopes,
- discrete-monitoring support,
- boundary differentiability at $S=B$,
- generalized rebate support,
- automatic promotion of every route-local decomposition into shared infrastructure.

Those exclusions are not weaknesses. They are part of the discipline that keeps analytical support honest.

# 11. Acceptance checklist for future analytical support

A route should be admitted into the shared analytical-support layer only if all items below are documented.

| Requirement | Question |
|---|---|
| Contract expression | Is the claim defined independently of its evaluator? |
| Model regime | Is the pricing model stated explicitly? |
| Validity envelope | Are all side conditions written down? |
| Kernel semantics | Does each kernel have a fixed semantic meaning? |
| Rewrite soundness | Is each decomposition justified as semantics-preserving? |
| Selector boundary | Is non-smooth dispatch outside the raw kernel core? |
| AD statement | Is the interior differentiability domain identified? |
| Testing plan | Can each kernel and rewrite be tested independently? |
| Promotion rationale | Is shared extraction justified by reuse or by independent semantic value? |

# 12. Recommended canonical vocabulary

For consistency across the analytical-support stack, the following vocabulary should be used canonically.

**Contract syntax**  
The structured claim expression.

**Valuation semantics**  
The mathematical value of the claim under a stated regime.

**Evaluator**  
The concrete method that computes that value.

**Kernel**  
A primitive valuation component with fixed semantic meaning and explicit validity envelope.

**Rewrite**  
A semantics-preserving replacement of one valuation target by an assembly of simpler components.

**Validity envelope**  
The full set of model and branch side conditions required for correctness.

**Smooth analytical core**  
The open-domain differentiable kernel layer.

**Public adapter / selector**  
The boundary layer containing dispatch, boundary handling, unsupported-feature logic, and fallbacks.

This vocabulary is sufficient to unify contract algebra, differentiable pricing, and implementation cookbook language without forcing premature generalization.

# 13. Recommended implementation obligations

Each analytical route package should ship the following artifacts.

1. A short contract-and-regime note.
2. A kernel semantic contract for every raw kernel.
3. A rewrite note or lemma list.
4. A selector note for branch routing and fallbacks.
5. A differentiability note for the raw interior domain.
6. Cross-checks against at least one independent engine where available.

This is the minimum documentary package required to keep analytical support auditable rather than mystical.

# 14. Final recommendation

The accessible source material already supports one strong architectural conclusion:

> Trellis analytical support should be formalized as contract syntax plus valuation semantics plus sound kernel rewrite, with a strict separation between raw differentiable kernels and the public selector / adapter boundary.

The barrier proof-of-concept is not just one successful formula extraction. It is the first clean proof that this vocabulary can support real route assembly without over-claiming a generic product algebra.

That makes it the right foundation for a broader contract-algebra document.

The next refinement step is not conceptual; it is editorial. Once the directly maintained documents named in your file map are added, this draft can be tightened into a shorter authoritative note with source-backed terminology for:

- basis-claim patterns,
- differentiable-pricing boundaries beyond the barrier case,
- the route cookbook layer contract,
- the exact current contract-algebra boundary.

# Appendix A. Formal notation summary

| Symbol | Meaning |
|---|---|
| $C$ | contract expression |
| $M$ | pricing model / regime |
| $x$ | market state / evaluator input |
| $\left[\!\left[ C \right]\!\right]_M(x)$ | semantic value of contract $C$ under model $M$ at state $x$ |
| $E_M(C,x)$ | evaluator output |
| $\Gamma$ | side-condition context / validity envelope |
| $K_i$ | semantically named kernel |
| $A$ | kernel assembly expression |
| $\rightsquigarrow$ | admissible semantics-preserving rewrite |

# Appendix B. Pseudocode interfaces

```text
raw kernel:
    value = vanilla_call_raw(params)
    value = barrier_image_raw(params)

selector:
    branch = barrier_regime_selector_raw(route_inputs)
    if branch == T09:
        return vanilla_call_raw(params) - barrier_image_raw(params)
    else:
        return fallback_engine(route_inputs)

semantic contract for a raw kernel:
    name: barrier_image_raw
    meaning: down-and-in image contribution on continuous-monitoring,
             zero-rebate, K > B Black-Scholes branch
    validity envelope:
        S > B
        K > B
        T > 0
        sigma > 0
        continuous monitoring
        zero rebate
```

# Appendix C. Mapping to the named work items

| Work item | Role in the present draft |
|---|---|
| `QUA-289` | template-plus-delta framing for analytical support |
| `QUA-291` | reusable analytical kernels |
| `QUA-292` | thin route interpreters over shared support |
| `QUA-293` | builder guidance for analytical assembly |
| `QUA-328` | concrete barrier proof-of-concept validating the pattern |

# Appendix D. Authority status of named documents

| Document name | Status in this draft |
|---|---|
| `qua-328-poc.md` / barrier proof-of-concept | directly consolidated through accessible barrier proof-of-concept material |
| `basis_claim_patterns.md` | used only as a directional anchor; not directly quoted |
| `differentiable_pricing.rst` | used only as a directional anchor; not directly quoted |
| `analytical_route_cookbook.rst` | used only as a directional anchor; not directly quoted |
| `contract_algebra.rst` | used only as a directional anchor; not directly quoted |
| `composition_calibration_design.md` | treated as historical background, not authority |
