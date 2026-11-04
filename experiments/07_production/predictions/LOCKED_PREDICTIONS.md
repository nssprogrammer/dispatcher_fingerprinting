# NB10 — locked predictions for the real-traffic (Chutes) analysis

Locked 2026-11-02, before any Chutes data was downloaded or seen. Detectors are frozen as fp-0.8 (classifier v0.7, spectral v0.8, hand-off graph v0.8, seed expansion as in NB9).

## Data and units
- Source: Chutes production LLM serving trace (HarvardMadSys/chutes_workload; Hugging Face `harvardMadsys/chutes_metadata`, CC BY 4.0). Per request: `user_id`, `rehash_round`, `instance_id`, `chute_id`, `started_at`, `completed_at`, `ttft`, token counts.
- Unit = account = (`user_id`, `rehash_round`). The trace has no API-key field, so an account plays the role of a key.
- Three 1-hour windows, chosen mechanically: the midpoint of shards at 20%, 50% and 80% of the sorted shard list. **W0 (the 50% shard) is the development window. W1 and W2 are held out.**
- Active account: ≥ 3 completed requests starting in the window. Strata: light (3–30 requests/h) and heavy (> 30/h).
- W0 may be used only to fix the activity floor, the stratum boundary, the injection placement and the duration source, and to debug code. If W0 exposes a pipeline bug, the fix and any changed prediction are recorded here *before* W1/W2 are opened.

## Predictions (apply to W1 and W2; W0 reported too)
- **R1 Real-traffic null, frozen classifier.** With the occupancy null re-estimated from the window's own random account pools (as in NB6–NB9), frozen v0.7 labels ≥ 95% of 200 random 40-account pools of *light* accounts as `none`, and ≥ 90% for *heavy* accounts.
- **R2 Surrogate calibration.** In random 40-account pools, the multi-scale ring statistic and successor concentration each exceed z = 6 against 40 time-shifted surrogates in ≤ 1% of pools.
- **R3 Replica coupling, real.** Pools of accounts with ≥ 90% of their requests on one serving instance have mean occupancy z within ±0.5 of random pools from the same stratum, and ≤ 5% beyond ±3 sd. If fewer than 3 instances host ≥ 40 such accounts, R3 is reported as not testable.
- **R4 Injection, classification.** Eleven synthetic pools, simulated with durations resampled from the window's own real requests, are added to the real background: shared counter, shuffle, per-worker counters, rotate-on-429 with a shared pointer and per worker (15/min quota), consistent hash, sticky 60 s, hedged, pinned, steady open loop, and a company router. Frozen v0.7 classifies ≥ 10 of 11 as in NB9.
- **R5 Injection, site-wide.** Spectral v0.8 run over all active accounts recovers the injected shared ring at recall ≥ 0.9 and purity ≥ 0.9. Any real accounts it selects form groups of their own and are not merged into the injected ring.
- **R6 Discoverability law on real noise.** For injected rings (no jitter, and 6 s jitter) against N ∈ {500, 2000, all active} real accounts, the shortest window in {5, 10, 15, 20, 30, 45, 60} min at which v0.8 recovers the ring (recall ≥ 0.9) is within a factor of 2 of T* = κ²Nτ_c/θ₁², using κ = 1.9, τ_c measured on that window's real accounts, and θ₁ from Proposition 1 with the pool's own durations. Holds in ≥ 75% of (ring, N, window) cells.
- **R7 Seed expansion against real background.** From a random 20% seed, label-free expansion ranks the rest of the injected shuffle, company, consistent-hash and hedged pools among all active accounts at AUC ≥ 0.85 within 20 minutes.
- **R8 Exploratory (no prediction).** Groups of *real* accounts found by spectral v0.8, the hand-off graph v0.8 and the per-pool classifier: their number, sizes and signatures, as a measured upper bound on the rate of cross-account shared dispatch (benign or not) in real traffic.

## Amendment 1 (pre-data, 2026-11-02, before NB10 was run): estimators for R4–R7
- Injected pools are simulated closed loops (client gap lognormal, median 0.2 s), with durations drawn i.i.d. from the window's own completed real requests and a 15/min per-key quota on every injected key, as in NB9. Pool sizes as in NB9: K = 8 over M = 40, except hedged (K = 6), pinned (K = M = 8), company (K = 4, M = 10), and the steady open loop (2 requests/s over 40 keys).
- R4 occupancy null: random 40-account pools from the stratum whose median requests per account is closest to the injected pools'.
- R6: τ_c = mean over the background accounts of fplib `key_corr_time` (integrated squared autocorrelation of 20 s high-pass occupancy at 0.5 s). θ₁ = Σ_d c(d) cos(2πd/M), with c(d) from Proposition 1 using the injected pool's own visit lags and durations (the NB6 profile check). Background = N accounts drawn at random from the window's active accounts. "Recovered" = spectral v0.8 on background + ring keys, restricted to [0, T), gives a group with recall ≥ 0.9 and purity ≥ 0.9.
- If the window has more active accounts than fit in memory for the dense detectors, "all active" becomes the largest feasible N, fixed on W0 and recorded here before W1/W2 are opened.

## Amendment 2 (after the W0 development pass, before W1 or W2 was opened; 2026-11-02)
Seen so far: W0 in full; for W1 and W2 only the extractor's descriptive profile (counts, duration and token percentiles, concurrency, instance counts — no detector output).

**Settings fixed on W0 (allowed by the lock):**
- Activity floor = 10 requests starting in the window. Strata: light 10–30, heavy 31–2,000, *platform* > 2,000 requests/h. Platform accounts (16 in W0, up to 410,049 requests/h) are excluded from random-pool tests and nulls (R1, R2, R4) and kept as background in every site-wide test (R5–R8). Reason: at floor 3 the grouping step chained 319 sparse accounts through chance overlaps; at floor 5 no group cleared the edge; at floors 10 and 20 the only group was the same 46 accounts. Platform accounts made per-pool statistics intractable (hundreds of millions of hand-off pairs per pool).
- Each analysis step runs in its own process. Injected pools use seed 1010 in every window.

**Implementation changes with identical output** (verified on NB7, NB8 B, NB9 A and B): hollowed correlation built by key chunks (max difference 9e-16); top eigenpairs by Lanczos when N > 1,500 (same selected keys and groups); hand-off counts accumulated in batches with bincount (identical counts).

**R3:** not testable in any window (no instance hosts ≥ 40 single-instance accounts).

**W0 outcomes (development, reported as such):** R1 pass (light 200/200, heavy 194/200 `none`); R2 light pass, heavy 2.5% ring z > 6 — accounted for by random pools that contain 4 or more members of a real 46-account group (expected 2.4% from membership counts; 0/40 pools with no member fired); R4 10/11; R5 fail; R6 4/6 cells (fail); R7 pass. The 46-account group: equal request counts (189–192/h, χ² = 0.10 on 45 df, p ≈ 8e-52), one model, frozen classifier "shared round-robin (open-loop relay)", ring z = 47.5.

**Unchanged:** R1–R8 as locked apply to W1 and W2.

**New predictions for W1 and W2, registered now:**
- **R2b.** Random pools with at most 3 members of any real group found by spectral v0.8 in the same window: ring z > 6 in ≤ 1%, in each stratum.
- **R6b.** With κ refitted on W0 (κ = 3.2, from the single detected cell: N = 500, T_det = 60 min, θ₁ = 2.73, τ_c = 5.29 s), R6's factor-of-2 criterion holds in ≥ 75% of (ring, N) cells.
- **R9.** Every group of real accounts found by spectral v0.8 that the frozen classifier labels a rotation (shared or per-worker) also has request counts per account more uniform than random dispatch allows (χ² lower-tail p < 0.01).

## Held-out outcomes (recorded after the frozen W1/W2 runs, 2026-11-02; code hashes in FROZEN_CODE.sha256)
- W1: R1 pass (light 200/200, heavy 199/200); R2 pass (0%); R2b pass; R4 10/11 pass; R5 fail; R6 fail (2/4); R6b pass (4/4); R7 pass (0.953/0.999/0.956/0.973); R9 not testable (no site-wide group).
- W2: R1 pass (light 199/200, heavy 192/200); R2 fail (heavy ring z>6 in 4%); R2b fail (4%; 2.6% counting the post-hoc 56-account group); R4 10/11 pass; R5 fail; R6 fail (4/6); R6b pass (6/6); R7 pass (0.943/0.999/0.965/0.959); R9 not testable.
- Post hoc (not predictions): heavy-only spectral v0.8 found a 56-account relay in W2 (chi2 lower-tail p 7e-34, ring z 42.0), a 12-account per-worker group in W1 (p 4e-6), and bunching groups of 16 (W2) and 8 (W1). Refitted law (kappa 3.2) on real relays: detection at 20/60/60 min vs T* 18/59/29 min; W2 site-wide T* 102 min, not found.
