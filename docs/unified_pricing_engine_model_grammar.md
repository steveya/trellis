# A Unified Mathematical and Computational Grammar for Calibration and Pricing Engines

## Purpose

This note turns the earlier mathematical synthesis into a concrete design
document for Trellis's latent-state and generator grammar.

It is the model-layer mathematical reference inside the broader Trellis
calibration architecture described in
`docs/mathematical/calibration.rst` and
`docs/developer/composition_calibration_design.md`.

The design goal here is not to replace the full Trellis calibration sleeve with
one separate master engine. The broader sleeve still includes market
reconstruction, runtime materialization onto `MarketState`, and later hybrid
composition. This note focuses on the common model-layer abstraction Trellis
uses when it needs a latent-state representation for model compression or
hybrid composition.

Within that model layer, the goal is to avoid building separate conceptual
engines for short-rate models, volatility models, credit models, and hybrids.
Instead, the model grammar should be built around a **single model-layer
abstraction** that can represent all of them and then dispatch to the right
numerical backend for pricing, Greeks, and calibration.

> **Framing note:**
> This document does not propose a separate calibration library outside
> Trellis. It also does not claim that every calibration workflow should end in
> one latent-state model. Curves, vol surfaces, credit curves, and correlation
> objects can be authoritative calibrated outputs in their own right and may
> later feed this model layer.

> **Working thesis:**
> The right unifier is not one master SDE for “short rate / variance / hazard rate.”  
> The right unifier is a **dynamic arbitrage-free pricing operator**, represented computationally by a latent stochastic state, its generator, a contract compiler, and a quote/calibration layer.

---

## 1. Executive summary

The core abstraction is

\[
Q_t(\eta) = \Psi_\eta\big(\Pi_t(H_\eta)\big),
\]

where:

- \(H_\eta\) is a family of claims indexed by contract coordinates \(\eta\)  
  (maturity, strike, tenor, recovery convention, etc.),
- \(\Pi_t\) is the no-arbitrage pricing operator,
- \(\Psi_\eta\) maps model prices into market quotes  
  (implied vol, par rate, CDS spread, hazard, etc.),
- \(Q_t(\eta)\) is the observable curve or surface.

A practical engine then represents \(\Pi_t\) through a latent state process \(X_t\) under a chosen numeraire / pricing measure, with generator \(\mathcal L_\theta\), and compiles each contract family into a standard pricing task:

\[
\partial_t u + \mathcal L_\theta u - c_\theta u + s_\eta = 0,
\qquad
u(T,\cdot;\eta)=g_\eta.
\]

This one equation already covers:

- rates through discounting \(c=r\),
- reduced-form default through killing \(c=r+\lambda\),
- coupons / recoveries / dividends through source terms \(s_\eta\),
- barriers / structural default through absorbing boundaries,
- option surfaces through quote transforms \(\Psi_\eta\),
- hybrids by enlarging the state vector and generator.

From an engine perspective, named models differ mainly by how they specify:

1. the latent state \(X_t\),
2. the generator \(\mathcal L_\theta\),
3. the potential term \(c_\theta\),
4. the contract family \(H_\eta\),
5. the quote map \(\Psi_\eta\),
6. the calibration objective.

That is the reusable abstraction.

---

## 2. Mathematical synthesis

### 2.1 Dynamic pricing operator

Let \(N_t\) be a numeraire and let \(\mathbb Q^N\) be the corresponding pricing measure. For any payoff \(H\) at time \(T\),

\[
\Pi_t^N(H) = N_t\,\mathbb E_t^{\mathbb Q^N}\!\left[\frac{H}{N_T}\right].
\]

This is the deepest financial object in the engine.

Everything market practitioners call a “curve” or “surface” is a family of evaluations of this operator:

- **yield curve:** evaluate \(\Pi_t\) on zero-coupon payoffs,
- **swap curve:** evaluate \(\Pi_t\) on floating and fixed legs, then convert to par rate,
- **implied vol surface:** evaluate \(\Pi_t\) on call/put payoffs, then invert the Black or Bachelier map,
- **survival curve / credit spread curve:** evaluate \(\Pi_t\) on default-sensitive cashflows, then convert to spreads or hazards.

So the true primitive is not “short rate,” “variance,” or “hazard.”  
The primitive is the pricing operator.

### 2.2 Latent-state representation

The pricing operator is too abstract to calibrate directly, so we represent it by a latent stochastic state \(X_t\in E\), parameterized by \(\theta\).

A broad template is

\[
dX_t = b_\theta(t,X_t)\,dt + \sigma_\theta(t,X_t)\,dW_t + dJ_t,
\]

or more generally a semimartingale characterized by drift, covariance, and jump measure.  
If the model is Markovian, it is equivalently specified by an infinitesimal generator

\[
\mathcal L_\theta f(x)
=
b_\theta(x)\cdot \nabla f(x)
+\frac12 \operatorname{tr}\!\big(a_\theta(x)\nabla^2 f(x)\big)
+
\int\!\Big(
f(x+\xi)-f(x)-\nabla f(x)\cdot h(\xi)
\Big)\nu_\theta(x,d\xi),
\]

with \(a_\theta=\sigma_\theta\sigma_\theta^\top\).

This is the key mathematical compression:

> a large fraction of the model zoo becomes “different parameterizations of the generator.”

### 2.3 Discounting, default, and running cashflows

For a contract with terminal payoff \(g_\eta(X_T)\), running source term \(s_\eta(t,X_t)\), and discount/default potential \(c_\theta(t,X_t)\), define

\[
u(t,x;\eta)
=
\mathbb E_{t,x}\!\left[
e^{-\int_t^T c_\theta(s,X_s)\,ds}g_\eta(X_T)
+
\int_t^T
e^{-\int_t^u c_\theta(s,X_s)\,ds}\,
s_\eta(u,X_u)\,du
\right].
\]

Then \(u\) solves the backward PDE/PIDE

\[
\partial_t u + \mathcal L_\theta u - c_\theta u + s_\eta = 0,
\qquad
u(T,\cdot;\eta)=g_\eta.
\]

This one template is the computational heart of the engine.

Interpretation of \(c_\theta\):

- \(r\) enters as discounting,
- \(\lambda\) enters as default killing,
- funding / convenience yields / mortality / other killing terms can be folded in the same way.

Interpretation of \(s_\eta\):

- coupons,
- dividends,
- recoveries,
- premium legs,
- running fees.

### 2.4 Default as killing or absorption

A useful unification is to treat reduced-form default as **killing** and structural default as **absorption**.

#### Reduced-form / intensity view

If \(\lambda_t=\lambda(X_t)\) is the intensity, then a survival-weighted claim has value

\[
u(t,x)
=
\mathbb E_{t,x}\!\left[
e^{-\int_t^T (r+\lambda)(s,X_s)\,ds}
g(X_T)
\right]
+\text{recovery/source terms}.
\]

Mathematically, rates and default simply add inside the potential term \(c=r+\lambda\).

#### Structural view

If default occurs when the state hits a set \(B\), then price is computed with an absorbing boundary condition at \(B\).  
Same engine; different event specification.

This is a much deeper unification than having one “rates engine” and one “credit engine.”

### 2.5 Curves and surfaces are observation maps

The market does not observe the latent state \(X_t\). It observes quotes indexed by contract parameters \(\eta\).

Define a family of claims \(H_\eta\). Then the quoted market object is

\[
Q_t(\eta)=\Psi_\eta\big(u(t,X_t;\eta)\big).
\]

Examples:

- **bond prices / yields / par rates:** \(\Psi\) converts prices to curve coordinates,
- **implied vols:** \(\Psi\) is Black/Bachelier inversion,
- **CDS spreads:** \(\Psi\) solves the par-spread equation for premium/protection legs.

This suggests a key engine rule:

> **Prices are core; quotes are transforms.**  
> Do not hard-code implied vol or par rate logic inside the model core.

### 2.6 Calibration as an inverse problem

Let \(\eta_1,\dots,\eta_n\) be the quoted contracts. Let

\[
Q^\theta = \big(Q^\theta(\eta_1),\dots,Q^\theta(\eta_n)\big)
\]

be the model-implied quote vector. Calibration is

\[
\theta^\star
=
\arg\min_{\theta}
\sum_{j=1}^n
w_j\,\ell\!\big(Q^\theta(\eta_j)-Q_j^{\mathrm{mkt}}\big)
+
R(\theta),
\]

where:

- \(w_j\) are weights,
- \(\ell\) is the misfit,
- \(R\) is regularization / prior / stability penalty.

This unifies:

- exact bootstrap as a special constrained fit,
- least-squares surface fitting,
- regularized curve smoothing,
- Bayesian calibration,
- joint cross-sectional and time-series estimation.

A useful geometric interpretation is:

> calibration is projection of the market quote vector onto the model-generated manifold in quote space.

That immediately explains:

- identifiability problems,
- sloppiness,
- unstable parameters,
- local minima,
- good cross-sectional fit but poor time dynamics.

### 2.7 Dynamic calibration and state estimation

Instead of recalibrating from scratch each day, one can fold calibration into a state-space system:

\[
X_{t+1}=F_\theta(X_t,\varepsilon_{t+1}),
\qquad
Y_t = h_\theta(X_t) + \epsilon_t,
\]

where \(Y_t\) is the observed quote vector or surface.

Then:

- one-shot calibration is static inversion,
- daily re-estimation is filtering,
- smoothing is dynamic data assimilation,
- uncertainty quantification becomes part of the framework.

This opens the door to:

- EKF / UKF for near-Gaussian models,
- particle filtering for nonlinear/non-Gaussian models,
- variational inference or MCMC when full posterior inference matters.

### 2.8 Infinite-dimensional market objects

Some of the natural state variables are not scalars but functions:

- forward-rate curve \(T \mapsto f_t(T)\),
- option price or implied-vol surface \((K,T)\mapsto C_t(K,T)\) or \(\sigma_t(K,T)\),
- survival curve \(T\mapsto S_t(T)\).

The mathematically clean statement is then

\[
X_t \in \mathcal H,
\]

for a function space \(\mathcal H\), with stochastic evolution in \(\mathcal H\).

This covers:

- HJM and other curve-dynamics models,
- direct surface models,
- rough/Volterra models via infinite-dimensional or lifted representations.

In production, this usually becomes

\[
\text{infinite-dimensional object}
\;\to\;
\text{finite-factor or lifted approximation}
\;\to\;
\text{pricing/calibration backend}.
\]

That reduction belongs in the compiler layer, not in the market-data layer.

---

## 3. The model grammar

The right engine grammar has three layers:

1. **Financial semantics:** numeraire, measure, cashflows, default/exercise events, quote conventions,
2. **Stochastic semantics:** latent state, dynamics, generator, boundary conditions,
3. **Numerical semantics:** solver choice, gradient method, regularization, optimizer/filter.

A named model is only one particular filling of this grammar.

### 3.1 Core abstract objects

A practical model specification can be built from the following abstract objects.

#### 1. Numeraire / measure

Choose a numeraire \(N\) and pricing measure \(\mathbb Q^N\).

Examples:

- money-market numeraire,
- \(T\)-forward numeraire,
- annuity numeraire.

This is not cosmetic; it can radically change numerical convenience.

#### 2. State space

\[
E = \mathbb R^d,\quad \mathbb R_+^d,\quad \mathcal H,\quad \text{or a lifted finite-dimensional approximation.}
\]

Examples:

- \(E=\mathbb R\times \mathbb R_+\) for Heston,
- \(E=\mathbb R_+\) for CIR intensity,
- \(E=\mathcal H\) for HJM,
- \(E=\mathbb R^{m}\) for a Volterra lift.

#### 3. Dynamics / generator

Specify \(X_t\) through one of:

- diffusion,
- jump-diffusion,
- affine process,
- regime-switching process,
- SPDE / function-space evolution,
- lifted approximation to a non-Markovian process.

The engine should compile these into either a generator \(\mathcal L_\theta\) or a simulation rule \(F_\theta\).

#### 4. Potential term

\[
c_\theta(t,x) = r_\theta(t,x) + \lambda_\theta(t,x) + \phi_\theta(t,x)+\cdots
\]

This collects discounting, default, funding, or other killing terms.

#### 5. Source term

\[
s_\eta(t,x)
\]

This collects running coupons, premium legs, recoveries, dividends, fees.

#### 6. Events and boundaries

Examples:

- absorbing default boundary,
- barrier event,
- early exercise,
- Bermudan schedule,
- callable / puttable features,
- coupon reset dates.

These must be first-class objects in the grammar.

#### 7. Contract family

A family of claims \(H_\eta\), indexed by contract coordinates \(\eta\).

Examples:

- \(H_{T}=1\) for zero-coupon bonds,
- \(H_{K,T}=(S_T-K)^+\) for European calls,
- CDS premium/protection legs indexed by maturity and coupon,
- swap legs indexed by tenor and fixed rate.

#### 8. Quote map

\[
\Psi_\eta:\text{price} \mapsto \text{market quote}
\]

Examples:

- price \(\mapsto\) Black implied vol,
- leg values \(\mapsto\) par swap rate,
- premium/protection legs \(\mapsto\) CDS spread.

#### 9. Calibration objective

A calibration spec should include:

- loss,
- weights,
- regularizer,
- hard constraints,
- priors,
- dynamic filtering logic if applicable.

#### 10. Backend hints

Examples:

- characteristic function / transform,
- PDE / PIDE,
- Monte Carlo / LSMC,
- low-rank tensor or sparse grid,
- surrogate or emulator,
- “auto” dispatch.

### 3.2 Pseudo-EBNF grammar

A useful abstract grammar is:

```text
EngineModel ::=
    NumeraireSpec
    MeasureSpec
    StateSpaceSpec
    StateVariableSpec
    DynamicsSpec
    PotentialSpec
    SourceSpec?
    EventSpec?
    BoundarySpec?
    ContractFamilySpec+
    QuoteMapSpec+
    CalibrationSpec
    BackendHints

StateSpaceSpec ::= FiniteDim(d)
                 | PositiveCone(d)
                 | FunctionSpace(H)
                 | LiftedVolterra(m)

DynamicsSpec ::= Diffusion(b, σ)
               | JumpDiffusion(b, σ, ν)
               | Affine(params)
               | RegimeSwitching(regimes, transitions)
               | SPDE(F, G)
               | LiftedKernel(weights, rates)

PotentialSpec ::= Discount(r)
                | Discount(r) + Default(λ)
                | Discount(r) + Default(λ) + Funding(φ)
                | Custom(c)

SourceSpec ::= Zero
             | RunningCashflow(s)
             | CouponSchedule(schedule)
             | Recovery(recovery_rule)
             | Composite(s_1 + ... + s_n)

EventSpec ::= None
            | Killing(λ)
            | AbsorbingBoundary(B)
            | Barrier(B)
            | Exercise(European | Bermudan(dates) | American)

ContractFamilySpec ::= TerminalPayoff(g_η, T)
                     | PathPayoff(F_η, [0,T])
                     | Defaultable(g_η, recovery, T)
                     | Portfolio(weights, families)

QuoteMapSpec ::= Price
               | ImpliedVol(Black | Normal)
               | ParRate
               | Spread
               | Hazard
               | Custom(Ψ)

CalibrationSpec ::= Objective(loss, weights)
                  + Regularizer(R)
                  + Constraints(C)
                  + EstimationMode(Static | Filtering | Bayesian)

BackendHints ::= Auto
               | Transform
               | PDE
               | PIDE
               | MonteCarlo
               | LSMC
               | Surrogate
```

### 3.3 Compiler target

The compiler should lower every model/instrument pair to a standard internal object:

```text
PricingTask = {
    measure,
    state_representation,
    generator_or_simulator,
    potential c(t,x),
    source s(t,x),
    terminal_condition g(x),
    path_state_augmentation,
    event/boundary rules,
    quote_map Ψ,
    gradient_request,
    backend_hints
}
```

This is the key architectural point:

> Backends should price `PricingTask`s, not “Heston,” “SABR,” or “CDS” directly.

### 3.4 Compilation semantics

The compilation pipeline should look like this:

1. **Normalize the contract**  
   Convert each instrument into terminal payoff, running cashflows, event rules, and quote convention.

2. **Choose measure / numeraire**  
   Use the most numerically convenient measure allowed by the contract family.

3. **Augment the state if needed**  
   Add running integrals, default indicators, barriers, coupon accrual states, or exercise states.

4. **Build the generator or simulator**  
   Compile diffusion / jump / lift / SPDE spec into a standard representation.

5. **Attach the potential and source terms**  
   Combine discounting, default killing, funding, coupons, recovery, etc.

6. **Select a backend**  
   Transform, PDE/PIDE, Monte Carlo, LSMC, surrogate, or hybrid.

7. **Solve for prices**  
   Produce prices for all relevant \(\eta\) on the quote grid.

8. **Map to market quotes**  
   Apply \(\Psi_\eta\) contract by contract.

9. **Differentiate**  
   Produce gradients / Jacobians / adjoints for calibration and Greeks.

10. **Optimize or filter**  
   Run static calibration, blockwise calibration, or dynamic state estimation.

---

## 4. Canonical engine interfaces

A clean software decomposition follows directly from the grammar.

### 4.1 Abstract interfaces

```python
class StateModel:
    state_space
    parameters
    def generator(self): ...
    def simulator(self): ...
    def parameter_transform(self): ...

class ContractFamily:
    parameter_space
    def compile_cashflows(self, eta): ...
    def compile_events(self, eta): ...
    def quote_map(self, eta): ...

class PricingTask:
    measure
    state_model
    terminal_payoff
    running_source
    potential
    events
    boundaries
    quote_map

class PricingBackend:
    def price(self, task, market_grid): ...
    def greeks(self, task, market_grid): ...
    def jacobian(self, task, calibration_params): ...

class Calibrator:
    objective
    regularizer
    constraints
    def fit(self, tasks, market_data): ...
```

### 4.2 Architectural principle

Keep these concerns separate:

- **financial meaning:** contract semantics and quote conventions,
- **model semantics:** latent state and dynamics,
- **numerics:** solver and differentiation,
- **inference:** optimization / filtering / Bayesian estimation.

Cross-asset engines usually become brittle when these layers are fused.

---

## 5. Example “compilations”

This section shows how named models become special cases of the grammar.

### 5.1 Heston stochastic volatility

#### Mathematical form

Let \(X_t=(x_t,v_t)\) with \(x_t=\log S_t\). Under the risk-neutral measure,

\[
dx_t = \left(r-q-\frac12 v_t\right)dt + \sqrt{v_t}\,dW_t^{(1)},
\]

\[
dv_t = \kappa(\theta-v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{(2)},
\qquad
d\langle W^{(1)},W^{(2)}\rangle_t = \rho\,dt.
\]

For European options, \(c=r\) and the terminal payoff is \(g_{K,T}(x,v)=(e^x-K)^+\).

#### Grammar instance

```yaml
name: Heston
numeraire: money_market
measure: risk_neutral
state_space: FiniteDim(2)
state_variables: [logS, variance]
dynamics:
  type: Diffusion
  drift:
    logS: r - q - 0.5 * v
    variance: kappa * (theta - v)
  covariance:
    - [v, rho * xi * v]
    - [rho * xi * v, xi^2 * v]
potential: r
source: 0
contracts:
  - family: EuropeanCall
    parameters: [K, T]
quote_map: BlackImpliedVol
calibration:
  parameters: [kappa, theta, xi, rho, v0]
  objective: vega_weighted_L2
backend_hints: Transform
```

#### Engine notes

- The semantic model is a 2D diffusion.
- The preferred backend is characteristic-function pricing / Fourier inversion.
- PDE is also viable because dimension is low.
- Implied vol is a quote transform, not part of the model itself.
- Parameter transforms should enforce \(v_0>0\), \(\theta>0\), \(\xi>0\), \(\rho\in(-1,1)\).

### 5.2 SABR

#### Mathematical form

For forward \(F_t\) and volatility \(\alpha_t\),

\[
dF_t = \alpha_t F_t^\beta\,dW_t^{(1)},
\qquad
d\alpha_t = \nu \alpha_t\,dW_t^{(2)},
\qquad
d\langle W^{(1)},W^{(2)}\rangle_t = \rho\,dt.
\]

#### Grammar instance

```yaml
name: SABR
numeraire: forward_numeraire
measure: T_forward
state_space: PositiveCone(2)
state_variables: [forward, alpha]
dynamics:
  type: Diffusion
  drift:
    forward: 0
    alpha: 0
  diffusion:
    forward_loading: alpha * F^beta
    alpha_loading: nu * alpha
    correlation: rho
potential: 0
source: 0
contracts:
  - family: EuropeanOptionOnForward
    parameters: [K, T]
quote_map: BlackOrNormalImpliedVol
calibration:
  parameters: [alpha0, beta, rho, nu]
  objective: weighted_surface_fit
backend_hints: Auto
```

#### Engine notes

- SABR is semantically just another 2D diffusion.
- The well-known Hagan formula should be treated as an **approximate backend**, not as the definition of the model.
- Depending on product and latency requirements, the engine can dispatch to:
  - asymptotic implied-vol approximation,
  - PDE,
  - Monte Carlo.
- Shifted variants and normal-lognormal switches belong in the quote map or contract convention layer.

### 5.3 CIR / short-rate family

#### Mathematical form

For a short rate \(r_t\),

\[
dr_t = \kappa(\theta-r_t)\,dt + \sigma\sqrt{r_t}\,dW_t.
\]

Bond prices are

\[
P(t,T)=\mathbb E_t\!\left[e^{-\int_t^T r_s\,ds}\right].
\]

This fits the generic PDE with potential \(c=r\), source \(0\), terminal payoff \(g\equiv 1\).

#### Grammar instance

```yaml
name: CIRShortRate
numeraire: money_market
measure: risk_neutral
state_space: PositiveCone(1)
state_variables: [short_rate]
dynamics:
  type: Diffusion
  drift:
    short_rate: kappa * (theta - r)
  diffusion:
    short_rate: sigma * sqrt(r)
potential: r
source: 0
contracts:
  - family: ZeroCouponBond
    parameters: [T]
  - family: Caplet
    parameters: [reset, payment, strike]
quote_map:
  - Price
  - Yield
  - ParRate
calibration:
  parameters: [kappa, theta, sigma, r0]
  objective: weighted_curve_plus_vol_fit
backend_hints: AffineTransform
```

#### Engine notes

- Rates are not a separate engine; they are the same PDE/semigroup engine with \(c=r\).
- Exact initial curve fit extensions such as shift models belong naturally in the observation or deterministic-shift layer.
- Swaptions add a quote map and contract family, not a new architecture.

### 5.4 Reduced-form credit intensity

#### Mathematical form

Let \(X_t=\lambda_t\) or a multi-factor extension. For example,

\[
d\lambda_t = \kappa(\theta-\lambda_t)\,dt + \sigma\sqrt{\lambda_t}\,dW_t.
\]

Survival is

\[
\mathbb Q(\tau>T\mid\mathcal F_t)
=
\mathbb E_t\!\left[e^{-\int_t^T \lambda_s\,ds}\right].
\]

A defaultable zero coupon bond under independent rates/intensity has price

\[
P^{\mathrm{def}}(t,T)
=
\mathbb E_t\!\left[e^{-\int_t^T (r_s+\lambda_s)\,ds}\right]
+
\text{recovery contribution}.
\]

#### Grammar instance

```yaml
name: ReducedFormCredit
numeraire: money_market
measure: risk_neutral
state_space: PositiveCone(1)
state_variables: [hazard]
dynamics:
  type: Diffusion
  drift:
    hazard: kappa * (theta - lambda)
  diffusion:
    hazard: sigma * sqrt(lambda)
potential: r + lambda
source:
  type: Recovery
  rule: fractional_recovery_of_par
contracts:
  - family: DefaultableBond
    parameters: [T, recovery]
  - family: CDS
    parameters: [maturity, coupon, accrual_convention]
quote_map:
  - Price
  - Spread
  - Hazard
calibration:
  parameters: [kappa, theta, sigma, lambda0]
  objective: spread_weighted_fit
backend_hints: AffineTransform
```

#### Engine notes

- Credit becomes almost identical to rates once default is recognized as killing.
- Premium and protection legs are just different source/terminal components.
- The same abstraction can also support structural credit by replacing intensity with absorbing boundary events.

### 5.5 HJM / function-space term-structure model

#### Mathematical form

Let the state be the entire forward-rate curve \(f_t(\cdot)\in\mathcal H\). In HJM form,

\[
df_t(T)=\mu_t(T)\,dt+\sum_{i=1}^m \sigma_i(t,T)\,dW_t^{(i)},
\]

with the HJM drift restriction ensuring no arbitrage.

#### Grammar instance

```yaml
name: HJM
numeraire: money_market
measure: risk_neutral
state_space: FunctionSpace(H)
state_variables: [forward_curve]
dynamics:
  type: SPDE
  drift: HJM_drift_restriction(sigma)
  diffusion: sigma(t, T)
potential: integral_of_short_rate_from_curve
source: 0
contracts:
  - family: ZeroCouponBond
    parameters: [T]
  - family: Swaption
    parameters: [expiry, tenor, strike]
quote_map:
  - Yield
  - ParRate
  - NormalOrLognormalVol
calibration:
  parameters: factor_loadings_or_kernel_params
  objective: curve_and_swaption_fit
backend_hints: FactorReduction
```

#### Engine notes

- This is the cleanest way to represent full curve dynamics.
- In practice the engine will compile the function-space model into:
  - factor truncation,
  - PCA basis,
  - finite-element basis,
  - low-rank kernel approximation.
- Again, same engine, larger state representation.

### 5.6 Hybrid equity-credit-stochastic-vol model

This is the type of case that exposes whether the abstraction is actually unified.

#### Mathematical form

Let

\[
X_t = (x_t,v_t,\lambda_t),
\]

with

\[
dx_t = \left(r-q-\frac12 v_t\right)dt + \sqrt{v_t}\,dW_t^{(1)},
\]

\[
dv_t = \kappa_v(\theta_v-v_t)\,dt+\xi_v\sqrt{v_t}\,dW_t^{(2)},
\]

\[
d\lambda_t = \kappa_\lambda(\theta_\lambda-\lambda_t)\,dt+\xi_\lambda\sqrt{\lambda_t}\,dW_t^{(3)},
\]

and a correlation matrix across the Brownian factors.

Default-sensitive equity derivatives can then be priced with potential \(c=r+\lambda\), plus any recovery or settlement rule.

#### Grammar instance

```yaml
name: EquityCreditHybrid
numeraire: money_market
measure: risk_neutral
state_space: PositiveCone(3)_with_log_equity
state_variables: [logS, variance, hazard]
dynamics:
  type: Diffusion
  drift:
    logS: r - q - 0.5 * v
    variance: kv * (thetav - v)
    hazard: kl * (thetal - lambda)
  covariance: full_correlated_CIR_Heston_block
potential: r + lambda
source:
  type: Custom
  rule: settlement_or_recovery_rule
contracts:
  - family: EquityOption
    parameters: [K, T]
  - family: CDS
    parameters: [maturity, coupon]
  - family: ConvertibleLikeOrHybridClaim
    parameters: [deal_specific]
quote_map:
  - BlackImpliedVol
  - Spread
calibration:
  parameters: [vol_params, credit_params, correlation_params]
  objective: joint_multi_surface_fit
backend_hints: Auto
```

#### Engine notes

- Nothing conceptually new is needed.
- Only the state dimension and correlation structure grew.
- The engine can dispatch to:
  - affine transform methods if the hybrid remains affine enough,
  - PDE if low-dimensional and simple enough,
  - Monte Carlo otherwise.

### 5.7 Structural default as an event, not a separate architecture

Suppose firm value \(V_t\) defaults when it hits a barrier \(B_t\). Then default time is

\[
\tau = \inf\{t : V_t \le B_t\}.
\]

The grammar becomes:

```yaml
name: StructuralCredit
state_variables: [firm_value, possibly_volatility, rates]
events:
  - type: AbsorbingBoundary
    set: V <= B(t)
contracts:
  - family: DefaultSensitiveClaim
quote_map:
  - Spread
  - BondPrice
backend_hints: PDE_or_MC
```

This shows how structural and reduced-form credit can live in one engine: one uses killing, the other absorption.

### 5.8 Rough or Volterra volatility through lifted states

A rough volatility model is often non-Markovian in its native form. In the engine, that should not force a new architecture. Instead:

- keep rough/Volterra semantics at the model layer,
- compile them into a lifted finite-dimensional Markov approximation,
- feed the lifted state into the same pricing task interface.

A grammar sketch is:

```yaml
name: RoughVolLift
state_space: LiftedVolterra(m)
dynamics:
  type: LiftedKernel
  kernel:
    weights: [w1, ..., wm]
    mean_reversion_rates: [a1, ..., am]
backend_hints: MC_or_PDE_if_small_m
```

The point is not the exact lift. The point is the abstraction: non-Markovian does not break the engine if the compiler can build a Markovian surrogate.

---

## 6. Numerical backends and dispatch

The same semantic model can support several numerical backends. This is a major reason to separate model semantics from numerics.

| Structure | Good backend(s) | Gradient method | Comments |
|---|---|---|---|
| Affine / transform-friendly | Fourier / COS / transform | AD through transforms or analytic derivatives | Best for Heston/CIR-class models |
| Low-dimensional diffusion | PDE / finite difference / finite element | adjoint PDE, AD | Strong for vanillas, barriers, short-rate models |
| Low-dimensional jump model | PIDE | adjoint PIDE, AD | Good when jumps matter and dimension is modest |
| Moderate/high-dimensional Markov | Monte Carlo | pathwise, likelihood ratio, AAD | Default workhorse for hybrids and path dependence |
| Optimal stopping | LSMC, PDE obstacle solver | AAD, differentiated regressions, adjoints | For Bermudans/Americans |
| Function-space model | factor reduction + PDE/MC | AD on reduced model | Compiler should reduce dimension |
| Rough / Volterra | lifted MC, lifted PDE if small | AAD on lift | Model stays unified through the lift |

### 6.1 Backend selection heuristics

A simple dispatch logic is:

1. **Exploit structure first**  
   If affine/transform structure exists, use it.

2. **Use PDE/PIDE when state dimension is low**  
   Usually best for dimensions up to about 2–3, sometimes 4 with strong sparsity or symmetry.

3. **Use Monte Carlo when dimension or path dependence dominates**  
   Especially for hybrids, callable structures, exposure simulation.

4. **Treat asymptotic formulas as optional approximations**  
   Useful for initialization, calibration speed, or sanity checks; not core semantics.

5. **Make pricing differentiable by design**  
   Backend choice should account for gradient stability, not just price speed.

### 6.2 Differentiable pricing matters

The engine is not just a pricing engine. It is a calibration engine.  
So the real production target is a **differentiable pricing program**.

For calibration, we need Jacobians such as

\[
\frac{\partial Q^\theta(\eta_j)}{\partial \theta_k}.
\]

Practical techniques include:

- adjoint PDE methods,
- algorithmic differentiation through transforms and root-finds,
- pathwise derivatives in Monte Carlo,
- likelihood ratio estimators,
- common random numbers for stable optimization,
- surrogate Jacobians when exact differentiation is too expensive.

A model with slightly worse raw pricing speed but much better gradient stability is often the better calibration model.

---

## 7. Calibration architecture

### 7.1 Unified objective

A general calibration objective should allow:

\[
\mathcal J(\theta)
=
\underbrace{\sum_j w_j\,\ell_j(Q^\theta_j-Q^{\mathrm{mkt}}_j)}_{\text{fit}}
+
\underbrace{R(\theta)}_{\text{regularization}}
+
\underbrace{C(\theta)}_{\text{hard/soft constraints}}
+
\underbrace{D(\theta,\theta_{\text{prev}})}_{\text{time stability / filtering}}.
\]

This covers:

- exact curve fit,
- smile/surface fit,
- joint rate-vol-credit fit,
- cross-day stability penalties,
- Bayesian priors,
- parameter smoothness across maturities or regimes.

### 7.2 Recommended calibration strategy

For a production engine, a robust pattern is:

1. **standardize the market quotes**  
   clean conventions, calendars, accruals, discounting assumptions;

2. **initialize structurally**  
   curve bootstrap, moment-based seed, asymptotic approximations;

3. **fit in blocks if natural**  
   e.g. rates first, then vol, then credit, then joint refinement;

4. **finish with a joint objective**  
   to reconcile cross-effects and correlations;

5. **stabilize over time**  
   with filtering or penalties on parameter changes.

### 7.3 Identifiability and sloppiness

Cross-asset calibration fails less often for mathematical reasons than for geometry reasons:

- multiple parameter vectors produce nearly identical quotes,
- some directions in parameter space are weakly informed,
- surface noise can drive parameter instability,
- daily recalibration can create fictitious dynamics.

Engine implications:

- regularization is not optional,
- parameter transforms and priors matter,
- one should expose condition numbers / Hessian diagnostics / sensitivity ranks,
- “calibration succeeded” should not mean only that the optimizer stopped.

### 7.4 Quote-space versus price-space fitting

The engine should support both:

- fitting in price space,
- fitting in implied-vol / spread / rate space,
- hybrid losses.

The quote map \(\Psi_\eta\) should expose both forward and inverse transforms so the calibration layer can choose the appropriate metric.

---

## 8. Arbitrage constraints and data normalization

The engine should not rely on the model alone to remove all market inconsistencies.  
A separate data layer should:

- normalize day count, calendars, accrual conventions,
- standardize discount curves and collateral assumptions,
- detect static arbitrage in surfaces when relevant,
- provide optional pre-smoothing or arbitrage projection,
- record quote uncertainties and liquidity weights.

This leads to an important design principle:

> A calibration engine should treat market data as noisy observations of an arbitrage-aware object, not as perfect truth.

For example:

- option surfaces may violate convexity/monotonicity because of noise,
- CDS curves may have inconsistent recovery assumptions,
- swaption cubes may mix quoting conventions,
- bond curves may contain stale data.

The model layer should not have to absorb all of this implicitly.

---

## 9. Implementation blueprint

A minimal but scalable engine architecture is:

```text
raw market data
    ↓
market-data normalization
    ↓
contract canonicalization
    ↓
model grammar / compiler
    ↓
pricing task(s)
    ↓
backend dispatch
    ↓
prices and gradients
    ↓
quote transformation
    ↓
objective / regularization / constraints
    ↓
optimizer or filter
    ↓
diagnostics / reports / stored state
```

### 9.1 Suggested modules

1. **Market data layer**  
   quotes, conventions, calendars, liquidity weights, static-arbitrage checks.

2. **Contract library**  
   bonds, swaps, swaptions, options, CDS, exotics, hybrid claims.

3. **Model grammar layer**  
   parsers / builders for state models, generators, potentials, sources, events.

4. **Compiler layer**  
   lowers model + contract families into `PricingTask`s.

5. **Pricing backends**  
   transform, PDE/PIDE, MC/LSMC, surrogate.

6. **Sensitivity layer**  
   Greeks, calibration Jacobians, adjoints, AAD.

7. **Calibration layer**  
   static optimizers, constrained fits, regularized fits, dynamic filters.

8. **Diagnostics layer**  
   residuals, stability, identifiability, arbitrage warnings, parameter history.

### 9.2 Parameter transforms and constraints

Parameters should never be stored only in “natural” model form.  
Use an unconstrained optimization parameter vector \(\phi\) and a transform \(\theta=T(\phi)\) that enforces:

- positivity,
- correlations in \((-1,1)\),
- Feller-like conditions if desired,
- monotonicity or shape constraints where relevant.

This makes optimization and filtering much more stable.

### 9.3 Caching and warm starts

A practical engine should cache:

- transform grids,
- PDE operators,
- compiled cashflow schedules,
- Monte Carlo paths or seeds where appropriate,
- previous-day solutions and Jacobians,
- implied-vol root-finding seeds.

Warm starts often dominate real calibration latency improvements.

---

## 10. Design principles for a unified engine

1. **Organize around pricing tasks, not named models.**

2. **Treat discounting, default, and funding uniformly through the potential term whenever possible.**

3. **Treat coupons, recoveries, fees, and running cashflows uniformly through source terms.**

4. **Treat curves and surfaces as observations of the pricing operator, not primitive state variables unless the model truly lives in function space.**

5. **Keep quote conventions outside the stochastic core.**

6. **Keep every backend differentiable or at least sensitivity-aware.**

7. **Let the compiler absorb complexity.**  
   Path dependence, default indicators, Volterra lifts, exercise states, and measure changes belong in compilation, not in ad hoc product code.

8. **Expose uncertainty and conditioning.**  
   Calibration quality is not just residual error.

9. **Allow several numerical realizations of the same model semantics.**  
   Heston may use Fourier today, PDE tomorrow, MC for a path-dependent claim.

10. **Treat time-series estimation as a first-class extension of calibration.**

---

## 11. What this buys you

If the engine is built this way, then adding a new model usually means adding only one of the following:

- a new state model / generator,
- a new contract family,
- a new quote map,
- a new backend plugin,
- a new calibration objective or regularizer.

It does **not** mean adding a brand new architecture for each asset class.

That is the real payoff of the abstraction.

---

## 12. A compact “model grammar” you can build around

A concise master representation is:

\[
\mathfrak M_\theta
=
\big(
N,\mathbb Q^N,
E,X_t,\mathcal L_\theta,
c_\theta,s_\eta,
\mathcal E,\mathcal B,
H_\eta,\Psi_\eta,
\mathcal J,
\mathcal N
\big),
\]

where:

- \(N,\mathbb Q^N\): numeraire and pricing measure,
- \(E,X_t,\mathcal L_\theta\): state space, state process, generator,
- \(c_\theta\): potential term,
- \(s_\eta\): source term,
- \(\mathcal E,\mathcal B\): events and boundaries,
- \(H_\eta\): contract family,
- \(\Psi_\eta\): quote map,
- \(\mathcal J\): calibration objective,
- \(\mathcal N\): numerical backend selection / hints.

A compiler maps this to a collection of standard pricing tasks.  
A calibrator maps model-implied quotes to market quotes.  
A filter extends it through time.

That is the unified engine blueprint.

---

## 13. Final recommendation

For a real calibration-and-pricing engine, I would recommend using the following as the **single central abstraction**:

\[
\boxed{
\partial_t u + \mathcal L_\theta u - c_\theta u + s_\eta = 0,
\qquad
Q^\theta(\eta)=\Psi_\eta(u),
\qquad
\theta^\star=\arg\min_\theta \mathcal J(Q^\theta,Q^{\mathrm{mkt}})
}
\]

with the following interpretation:

- \(\mathcal L_\theta\): latent stochastic dynamics,
- \(c_\theta\): discount/default/funding potential,
- \(s_\eta\): running cashflow / recovery / source,
- \(u\): contract value,
- \(\Psi_\eta\): quote transform,
- \(\mathcal J\): calibration / inference objective.

Then:

- SABR, Heston, CIR, HJM, reduced-form credit, structural credit, and hybrids all become grammar instances,
- transform, PDE, PIDE, MC, and lifted approximations become numerical backends,
- calibration, filtering, and Bayesian estimation become alternative inference modes on the same architecture.

That is the right level of abstraction if the goal is to build one durable computational engine instead of a set of disconnected model-specific calibrators.
