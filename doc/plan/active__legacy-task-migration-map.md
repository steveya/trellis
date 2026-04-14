# Legacy Task Migration Map

This table is the audit output from the former `TASKS.yaml` corpus. The legacy manifest has been replaced by explicit benchmark, extension, negative, and proof corpora, and this map remains as the migration record.

| Task | Bucket | Target | Title |
| --- | --- | --- | --- |
| `T01` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical |
| `T02` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Callable bond: BDT lognormal vs HW normal tree |
| `T03` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | Trinomial tree implementation and convergence |
| `T04` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Bermudan swaption on HW tree |
| `T05` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Puttable bond: exercise_fn=max and puttable-callable symmetry |
| `T06` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CIR++ rate tree: positive rates via shifted CIR |
| `T07` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Two-factor Hull-White tree (2D lattice) |
| `T08` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Convertible bond: equity + credit on tree |
| `T09` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Step-up callable bond (varying coupon schedule) |
| `T10` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | Tree convergence study: price oscillation and Richardson extrapolation |
| `T11` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Lattice Greeks via pathwise differentiation on tree |
| `T12` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Range accrual note on rate tree |
| `T13` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | European call: theta-method convergence order measurement |
| `T14` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | American put: PSOR vs tree vs LSM three-way |
| `T15` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CEV model: CEVOperator PDE vs CEV tree |
| `T16` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Barrier call: PDE absorbing BC vs MC discrete monitoring |
| `T17` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Callable bond: HW rate PDE (PSOR) vs HW tree |
| `T18` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Log-space PDE for rate instruments (avoid negative rates) |
| `T19` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Non-uniform grid (log-spaced) for PDE near-barrier accuracy |
| `T20` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | 2D PDE: Heston (S, V) via ADI splitting |
| `T21` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | European put PDE: put-call parity verification |
| `T22` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Double barrier option via PDE |
| `T23` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Crank-Nicolson Rannacher smoothing for discontinuous payoffs |
| `T24` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Finite element method (FEM) vs finite difference for European |
| `T25` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | GBM call: all 4 schemes convergence order |
| `T26` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Variance reduction: antithetic + control variate + importance sampling |
| `T27` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | LSM basis function shootout at σ=0.20 and σ=0.40 |
| `T28` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Heston MC: Euler vs QE scheme (Andersen 2008) |
| `T29` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Asian option (arithmetic average): MC vs Turnbull-Wakeman |
| `T30` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Lookback option: MC vs analytical (Goldman-Sosin-Gatto) |
| `T31` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | Quasi-Monte Carlo: Sobol vs pseudo-random convergence |
| `T32` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Multi-level Monte Carlo (MLMC) for variance reduction |
| `T33` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Merton jump-diffusion MC vs FFT |
| `T34` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | SABR MC simulation vs Hagan implied vol |
| `T35` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Basket option (2 assets): MC with Cholesky correlation |
| `T36` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Autocallable note: MC path-dependent with early redemption |
| `T37` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Variance swap: MC replication vs analytical (log contract) |
| `T38` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | CDS pricing: hazard rate MC vs survival prob analytical |
| `T39` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | FFT vs COS: GBM calls/puts across strikes and maturities |
| `T40` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Heston smile: FFT vs COS vs MC implied vol surface |
| `T41` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Variance Gamma: COS vs MC |
| `T42` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | COS adaptive truncation for extreme parameters |
| `T43` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Merton jump-diffusion: FFT vs COS vs MC |
| `T44` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Bates model (Heston + jumps): FFT vs MC |
| `T45` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CGMY / tempered stable process via COS |
| `T46` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Digital option pricing via FFT and COS |
| `T47` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Spread option via 2D FFT (Hurd-Zhou) |
| `T48` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Characteristic function registry: unify GBM/Heston/VG/Merton/Bates |
| `T49` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CDO tranche: Gaussian vs Student-t copula |
| `T50` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Nth-to-default: MC correlated defaults vs semi-analytical |
| `T51` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | CDS par spread: hazard rate bootstrap vs closed-form |
| `T52` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CVA on interest rate swap: MC exposure simulation |
| `T53` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Multi-name portfolio loss distribution: recursive vs FFT vs MC |
| `T54` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Wrong-way risk: correlated default and exposure |
| `T55` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Credit index option: Black on spread vs MC |
| `T56` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Vasicek bond pricing: tree vs analytical |
| `T57` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CIR bond pricing: tree vs analytical |
| `T58` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Kou double-exponential jump: FFT vs MC |
| `T59` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Local volatility: Dupire PDE forward vs MC with local vol |
| `T60` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Stochastic local vol (SLV): Heston + local vol mixing |
| `T61` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Mean-reverting commodity model (Schwartz 1-factor) |
| `T62` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | G2++ two-factor rate model |
| `T63` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | SABR MC: dynamic vs static smile |
| `T64` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Hull-White with time-dependent vol: piecewise constant sigma(t) |
| `T65` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | HW calibration to swaption volatilities |
| `T66` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | SABR calibration to smile at multiple expiries |
| `T67` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | Heston calibration to vanilla option surface |
| `T68` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Local vol surface construction from market IVs |
| `T69` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | BDT calibration to yield vol term structure |
| `T70` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | Bootstrap yield curve from swap rates |
| `T71` | `market_or_research_hold` | `TASKS_PROOF_LEGACY.yaml` | CDS hazard rate bootstrap from spreads |
| `T72` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Implied vol surface: SVI parameterization |
| `T73` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | European swaption: Black76 vs HW tree vs HW MC |
| `T74` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | European equity call: 5-way (tree, PDE, MC, FFT, COS) |
| `T75` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | American put: tree vs PDE vs LSM at 3 vol levels |
| `T76` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Heston European: analytical vs MC vs PDE vs FFT vs COS |
| `T77` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Cap/floor: Black76 vs HW tree vs MC rate simulation |
| `T78` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Barrier option: PDE vs MC vs tree vs analytical |
| `T79` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Convertible bond: tree vs MC vs PDE |
| `T80` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Bermudan swaption: tree vs LSM MC |
| `T81` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Bond Greeks: autograd vs FD vs analytical |
| `T82` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Callable bond full analytics: vega, OAS, duration, scenarios |
| `T83` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | KRD: interpolation-aware bumping |
| `T84` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Scenario analysis: parallel + twist + butterfly |
| `T85` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | YTM solver for bonds |
| `T86` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Theta (time decay) for options via tree and PDE |
| `T87` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Rho (rate sensitivity) for equity options |
| `T88` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Book P&L attribution: rate + spread + vol decomposition |
| `T89` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | OAS duration (spread duration) for callable bonds |
| `T90` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Vega surface: per-expiry per-strike vega bucketing |
| `T94` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | FX market bridge: Garman-Kohlhagen vs MC with explicit domestic/foreign curve selection |
| `T95` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | xVA framework: CVA + DVA + FVA on IR swap portfolio |
| `T96` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Lookback option (fixed strike): MC vs Goldman-Sosin-Gatto analytical |
| `T97` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Digital (cash-or-nothing) option: BS formula vs MC vs COS |
| `T98` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Geometric Asian option: closed-form vs MC |
| `T99` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Chooser option: Rubinstein formula vs MC |
| `T100` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Compound option: Geske formula vs MC |
| `T101` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Cliquet option: forward-start decomposition vs MC |
| `T102` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Rainbow option (best-of-two): Stulz formula vs MC |
| `T103` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Double barrier option: Ikeda-Kunitomo vs PDE vs MC |
| `T104` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Variance swap: log-contract replication vs MC realized var |
| `T105` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Quanto option: quanto-adjusted BS vs MC cross-currency |
| `T106` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Forward start option: BS closed-form vs MC |
| `T107` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Autocallable note: MC with barrier + coupon + early redemption |
| `T108` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | FX vanilla option: Garman-Kohlhagen vs MC |
| `T109` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | FX barrier option: analytical vs MC |
| `T110` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | FX digital (one-touch): Reiner-Rubinstein vs MC |
| `T111` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | FX double barrier: analytical vs PDE vs MC |
| `T112` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | FX variance swap: replication vs MC |
| `T113` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Gauss-Hermite quadrature for swaption pricing |
| `T114` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Gauss-Laguerre integration for Heston (alternative to FFT) |
| `T115` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CONV method for Bermudan under Variance Gamma |
| `T116` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Rough Bergomi: hybrid simulation vs rough Heston semi-analytical |
| `T117` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Local-stochastic vol (LSV): 2D PDE vs MC with leverage function |
| `T118` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Chebyshev spectral method vs FD for European option |
| `T119` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | NIG process: COS pricing vs MC |
| `T120` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Margrabe exchange option: closed-form vs 2D MC |
| `T121` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | Convertible bond: Tsiveriotis-Fernandes tree vs MC |
| `T122` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Inflation-linked bond: real yield curve pricing |
| `T123` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | CMS cap: convexity adjustment (Hagan) vs replication vs MC |
| `T124` | `benchmark_rewrite_candidate` | `rewrite/new corpus` | CDS option: Black76 on spread vs MC |
| `T125` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Swing option: dynamic programming on tree vs LSM MC |
| `T126` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Spread option (Kirk approximation) vs 2D MC vs 2D FFT |
| `E21` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | European equity call: 5-way (tree, PDE, MC, FFT, COS) |
| `E22` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Cap/floor: Black caplet stack vs MC rate simulation |
| `E23` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | European equity call under local vol: PDE vs MC |
| `E24` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Merton jump-diffusion call: MC vs FFT |
| `E25` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | FX option (EURUSD): GK analytical vs MC |
| `E26` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | Nth-to-default basket: Gaussian copula vs default-time MC |
| `E27` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | American Asian barrier under Heston: PDE vs MC vs FFT should block honestly |
| `E28` | `proof_only_hold` | `TASKS_PROOF_LEGACY.yaml` | European equity call: transform-family separation (FFT vs COS) |
