# `pit_robust_betas` parallelism: measurements and decisions

Scope: how `pit_robust_betas` parallelizes its per-date Huber-RLM fits, what that
costs in time and memory at production width, and why both parallel axes ship.

All figures below are **measured** unless marked extrapolated. Synthetic panels
throughout (`scratch/pit_betas_bench/bench_common.py:make_panel` in
`dagster-pipelines-internal`); real panels are raggeder, and raggedness changes
load balance on both axes. Measured on WSL2, 12 cores, 15 GB RAM, joblib 1.5.1.
Ratios travel between machines; absolute numbers do not.

Related: [pit-betas-date-axis-plan.md](pit-betas-date-axis-plan.md) covers the
date-axis implementation and the serial rules it has to reproduce.

## 1. The production shape

The heaviest caller is the `dagster-pipelines-internal` market betas stage
(`betas_pipeline.py`), which passes `rebalance_dates = input_returns.index` --
every timestamp is a rebalance date, so the date loop runs `T` times.

| | value | source |
|---|---|---|
| T | 1393 | ArcticDB `us/stocks/1d` symbol metadata, 2020-12-31 .. 2026-07-21 |
| N | ~21000 | same (`rets` 21183 cols, `rets_figi` 20763) |
| K | 1 | SPY |
| `n_jobs` | 6 | `BETAS_PARALLEL_JOBS` |

One full `(T, N)` float64 frame is 234 MB.

## 2. joblib auto-memmapping exhausted the spool filesystem

`pit_robust_betas` holds **one** `joblib.Parallel` for the whole date loop, and
`compute_betas_fast` ships a `(window, chunk)` slice per task. joblib memmaps any
task array over `max_nbytes` (default 1 MB) into a file under `/dev/shm` (or
`$TMPDIR`), and those files are reclaimed **only when the pool exits**. The spool
therefore grew as the cumulative sum of shipped bytes rather than staying at the
few slices actually in flight.

Measured, N=21000, `n_jobs=6`, spool sampled after each date:

| date | window t | spool MB | cumulative shipped MB |
|---|---|---|---|
| 25 | 88 | 26 | 246 |
| 100 | 163 | 1603 | 1823 |
| 200 | 263 | 5190 | 5410 |
| 237 | 300 | 6943 | 7163 |

The spool tracks cumulative shipped bytes almost exactly and falls to zero only
at pool exit. Over a full run that total is `N*8*T^2/2` -- **~163 GB** at
T=1393 N=21000. A full-width benchmark run died at roughly date 425 with
`OSError: [Errno 28] No space left on device` after filling a 7.9 GB `/dev/shm`.

**Fix:** `Parallel(..., max_nbytes=None)` on the date-loop pool, which pickles
slices inline and frees them with their task. Spool held at 0 MB at every
checkpoint. Cost, same 237 dates:

| | spool at date 237 | fit seconds |
|---|---|---|
| joblib default | 6943 MB | 286.0 |
| `max_nbytes=None` | 0 MB | 295.0 |

**+3.1% wall clock to remove ~163 GB of spool.**

Diagnostic note, for anything reported as "memory grows through a betas run":
tmpfs pages are memory, but they appear in neither `tracemalloc` (not Python
heap) nor any process's RSS. A leak hunt using either tool finds nothing and
concludes there is no growth.

## 3. Per-date transient on the asset axis

Parent-side allocation per rebalance date, T=1393 N=21000 K=1, `n_jobs=6`, from
`tracemalloc` around a single `compute_betas_fast` call:

| window t | live cols | t*N*8 MB | parent peak MB | x panel slice |
|---|---|---|---|---|
| 199 | 9097 | 14 | 31 | 2.13 |
| 597 | 21000 | 100 | 210 | 2.10 |
| 995 | 21000 | 167 | 349 | 2.09 |
| 1393 | 21000 | 234 | **488** | 2.08 |

Dead linear at **2.09x** the live window slice, holding across a 100x range of
window bytes. Attribution: the full `(t, N)` `y_weighted` copy (1.00x), the
`y_weighted[:, ix]` chunk slices joblib pre-dispatches (1.00x), and `xw` /
`complete_rows` / `out` (0.09x).

This is allocated **and freed** every date, so it is a sawtooth of growing
amplitude, not a ratchet. It is not a leak: under `tracemalloc` a serial run held
live Python bytes flat at ~25 MB from date 16 to date 999.

## 4. Post-loop stage

The reindex, per-factor ffill, shift, hedge multiply and groupby-sum, at
T=1393 N=21000 K=1:

| metric | value |
|---|---|
| tracemalloc peak | 1405 MB |
| the 4 returned frames, alive at once | 936 MB |
| wall | 4.6 s |

`df_hedge_rets_by_fact` is `(T*K, N)` -- the largest frame the function builds --
and callers that want only betas and residuals discard it. `return_hedge_rets_by_fact=False`
frees it as soon as `df_hedge_rets` has been summed from it:

| `return_hedge_rets_by_fact` | peak MB | held MB |
|---|---|---|
| True | 1405 | 936 |
| False | 1405 | 702 |

**Peak is unchanged.** The peak occurs earlier, in the ffill/shift/multiply chain
where each step materializes its own frame; freeing the panel after that peak has
passed cannot lower it. The saving is 234 MB (25%) of *held* memory after the
return. Lowering the peak would require accumulating `df_hedge_rets` per factor
instead of materializing the by-factor panel at all -- not implemented.

## 5. Parallel axis: asset-level vs date-level

The asset axis parallelizes *inside* each date (one task per block of assets),
paying one fan-out and barrier per rebalance date. The date axis fans out over
dates: panel written once and memmapped read-only into every worker, one barrier
for the whole run, per-task payload a list of ints.

12 workers, K=3, every timestamp a rebalance date:

| config | variant | wall s | peak PSS MB | speedup | memory |
|---|---|---|---|---|---|
| T=500 N=100 | asset axis | 11.39 | 1076 | 1.00x | 1.00x |
| | date axis | 4.85 | 1069 | 2.35x | 0.99x |
| T=1000 N=200 | asset axis | 48.41 | 1096 | 1.00x | 1.00x |
| | date axis | 29.59 | 1084 | 1.64x | 0.99x |
| T=2000 N=400 | asset axis | 314.67 | 1145 | 1.00x | 1.00x |
| | date axis | 244.21 | 1129 | 1.29x | 0.99x |

At the production shape, K=1, `n_jobs=6`, both variants carrying the
`max_nbytes` fix:

| config | variant | wall s | fit s | peak RSS | peak PSS | speedup | memory |
|---|---|---|---|---|---|---|---|
| T=1393 N=21000 K=1 | asset axis | 6129.3 | 6125.6 | 3343 | 2519 | 1.00x | 1.00x |
| | date axis | 5371.5 | 5360.1 | 5528 | **2794** | **1.14x** | **1.11x** |

Read PSS, not RSS. RSS charges the shared memmapped panel to every worker that
maps it, which overstates the date axis's cost. PSS splits shared pages and is
the physical cost to the machine.

**The date axis is faster, and the margin shrinks as panels widen.** 2.35x at
N=100, 1.29x at N=400, **1.14x at N=21000**. The advantage is the *barrier
count*: `T` fan-outs versus one. A date carrying 100 fits cannot amortize its own
fan-out, so removing it is worth 2.35x; a date carrying 21000 fits amortizes it
almost completely, so removing it is worth 1.14x. Memory is a wash at narrow
widths and **11% worse** at production width.

### Outcome

Both axes ship. `parallel_axis` selects between them when `parallel=True` and
defaults to `"date"`. `parallel=False` runs the serial path and ignores the
argument.

| path | selected by |
|---|---|
| serial | `parallel=False` |
| date axis | `parallel=True, parallel_axis="date"` (default) |
| asset axis | `parallel=True, parallel_axis="asset"` |

All three return identical betas.
`tests/stats/test_pit_robust_betas_date_axis_equiv.py` and
`tests/stats/test_pit_robust_betas_parallel_equiv.py` assert bit-identity against
the serial path, so the choice affects throughput and memory only.

Two consequences of the default:

* A caller passing `parallel=True` and nothing else runs on the date axis, and at
  production width takes the 11% peak-memory increase with it. The source of that
  increase has not been diagnosed; the candidates are recorded in
  [pit-betas-date-axis-plan.md](pit-betas-date-axis-plan.md) §7. A caller
  optimizing for memory on a very wide panel should pass `parallel_axis="asset"`.
* `dagster-pipelines-internal` reads the axis from `BETAS_PARALLEL_AXIS`, so
  production can revert to the asset axis without a release.

The narrow-panel case is where the date axis pays. At hundreds rather than tens
of thousands of assets the measured speedup is 1.6-2.4x.

### Not evaluated here

Truncating the EWMA lookback -- dropping the negligible tail of the exponential
decay -- is a much larger speedup than any axis change, and it **changes results**.
It needs its own accuracy analysis and is deliberately not bundled with
parallelism work.
