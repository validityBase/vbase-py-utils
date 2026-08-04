# Plan: date-level parallel axis for `pit_robust_betas`, selectable and default

**Goal.** Add `parallel_axis: Literal["date", "asset"] = "date"` to
`pit_robust_betas`. The date axis becomes the default parallel path; the asset
axis stays reachable and unchanged; `parallel=False` stays the independent serial
reference.

**Status.** In progress. Reference prototype exists and is measured; the port is
being written against it rather than from it.

## 0. Read first: this is chosen against the measured trade

Measured at the production shape (T=1393, N=21000, K=1, 6 jobs), both variants on
current code -- see [pit-betas-parallelism.md](pit-betas-parallelism.md):

| variant | wall s | peak PSS MB |
|---|---|---|
| asset axis | 6129.3 | 2519 |
| date axis | 5371.5 | 2794 |

**1.14x faster, 11% more peak memory.** The speedup decays with panel width
(2.35x at N=100, 1.64x at N=200, 1.29x at N=400, 1.14x at N=21000) because the
date axis's advantage is barrier count, and a wide date amortizes its own fan-out.

Two consequences the plan carries rather than hides:

* Making the date axis the **default** is a memory regression for the widest
  caller. That regression is **accepted, not closed**: the diagnosis in Phase 4 is
  deliberately deferred, so `dagster-pipelines-internal` picks up +275 MB peak PSS
  on upgrade without a code change. This must be stated in the PR body, and
  `BETAS_PARALLEL_AXIS` (§9) is the production fallback if it bites.
* The narrow-panel case is where this pays. A caller with hundreds rather than
  tens of thousands of assets sees 1.6-2.4x, and that is the justification to name
  in the PR.

## 1. Parameter surface

```python
def pit_robust_betas(
    ...,
    parallel: bool = False,
    parallel_axis: Literal["date", "asset"] = "date",
    n_jobs: int = -1,
    blocks_per_worker: int = 4,
    ...
)
```

* `parallel_axis` is meaningful only when `parallel=True`. When `parallel=False`
  the existing serial `robust_betas` path runs unchanged, and `parallel_axis` is
  ignored -- do **not** route serial through the date code, or `parallel=False`
  stops being the independent reference the equivalence tests compare against.
* Reject an unknown `parallel_axis` value with a `ValueError` naming the input, in
  the same style as the existing duplicate-name guards. Do not silently fall back.
* `blocks_per_worker` controls both load balance and the size of the result
  payload each task returns; see 4.4.

Callers that must keep the asset axis pass `parallel_axis="asset"`. Nothing in
this repo forces the choice on them.

## 2. Repo topology

The fit code is here (`vbase-py-utils`). The reference prototype lives in
`dagster-pipelines-internal` at `scratch/pit_betas_bench/`:

| file | role |
|---|---|
| `_date_worker.py` | worker-side fits; import graph is numpy + `_huber_rlm` only |
| `date_parallel_lean.py` | orchestrator: precompute, panel spill, pool, streaming sink |
| `bench.py` | `--check` equivalence, `--sizes` throughput sweep |

The prototype is a **standalone drop-in** that re-implements all of
`pit_robust_betas`, including validation and post-processing. Do not port it that
way -- see Phase 3.

## 3. Port the modules (Phase 1)

**3.1 `vbase_utils/stats/_date_worker.py`** -- from the prototype's
`_date_worker.py`. Keep the import discipline: numpy plus
`vbase_utils.stats._huber_rlm`, nothing else. A worker that maps pandas costs
109 MB Pss against 77 MB for one that does not; the first cut of the prototype
broke this by importing a constant from `robust_betas`, which pulls pandas in
behind it.

Note that with §5.2 resolved as "replicate serial exactly", **every variance
check moves to the parent**, so the worker no longer needs
`NEAR_ZERO_VARIANCE_THRESHOLD` at all. The duplicated constant goes away rather
than being carried across.

**3.2 `vbase_utils/stats/_date_betas.py`** -- from `date_parallel_lean.py`, but
only `precompute`, `_write_block`, `betas_date_parallel` and `betas_serial`. It
exposes one entry point that returns the filled `betas_buf` and nothing else.

`betas_serial` is **not** optional. `joblib.Parallel(n_jobs=1)` selects the
sequential backend, which does not run `initializer`, so `_G["pc"]` is never
populated and `date_block` raises `KeyError`. Route `effective_n_jobs(...) == 1`
to the serial-over-dates loop.

**3.3 Integrate as a fit-stage replacement, not a parallel implementation.**
`pit_robust_betas` keeps ownership of every input check, the betas frame
construction, the reindex, the per-factor ffill and shift, the hedge arithmetic
and the return dict. Only the `sim()` call is swapped:

```python
if parallel and parallel_axis == "date":
    betas_buf = compute_betas_buf_by_date(...)   # fills the same buffer
else:
    sim(*sim_args, progress=progress, on_result=write_betas)
```

The prototype's duplicated post-processing block (its lines 230-273) is
**discarded**. Duplicating it would fork the cross-factor ffill/shift fix, the
`return_hedge_rets_by_fact` handling and the reindex skip -- all of which have
already cost a defect each.

## 4. Parity items the prototype does not have (Phase 2)

**4.1 Progress reporting.** `sim()` drives a tqdm bar and `calculate_betas` in
`dagster-pipelines-internal` passes `progress=True`. The date path has none.
Wrap the `return_as="generator_unordered"` consumption in tqdm over completed
blocks. Note the granularity necessarily changes from dates to blocks.

**4.2 Worker logging.** `_fast_betas._init_worker` sets `VBASE_LOG_LEVEL`, installs
a stderr handler so loky forwards worker records to the parent, and warms the JIT.
`init_dates` does only the BLAS pinning and the JIT warm. Port the logging setup
or worker-side diagnostics vanish silently.

**4.3 Per-fit labels.** The asset path calls
`fit_huber_rlm_params(y_f, x_c, label=col)`, which is what makes "Perfect fit for
<asset>" (`_huber_rlm.py:185`) actionable. The prototype passes no label. This is
not a trade-off to accept: ship the asset names to the workers once through
`initargs` and index them by the `cols` array the fit already has.

**4.4 Result payload size.** Each task returns `(row, cols, params)` per date with
`params` shaped `(n_facts, n_cols)`. At K=1, N=21000 that is ~168 KB per date;
with `blocks_per_worker=4` and 6 jobs, a block spans ~58 dates and returns ~10 MB.
That is fine, but it scales with `blocks_per_worker` and K, and it travels back
through the same pickling path whose input side just caused an ENOSPC failure.
Size it deliberately and say so in a comment.

**4.5 The per-window warnings.** Three fire on the serial/asset path and none on
the date path. All three dedupe to one WARNING per process, so reinstating them in
the parent is cheap:

| warning | serial condition |
|---|---|
| `_warn_incomplete_factors` | window is missing a factor (`i < all_facts_live_from`) |
| `_warn_no_asset_columns` | no asset has listed yet at this date |
| `_warn_deleted_rows` | complete-case deletion dropped rows from the window |

The exact firing population matters for the first two: `sim()` skips the callback
entirely when **every** masked object is empty, i.e. when there are neither live
assets nor live factors, so no warning is emitted for those dates.

**4.6 Decay validation.** `exponential_weights` validates `half_life > 0` and
`0 < lambda_ < 1` and raises when neither is given. The prototype's `precompute`
derives `lambda_ = exp(log(0.5)/half_life)` with no checks, so a bad input yields
garbage betas instead of a `ValueError`. Extract the resolve-and-validate step
into one helper in `robust_betas` and call it from both.

## 5. Defects in the prototype -- fixed before porting

**5.1 Rebalance timestamps before the panel start are handled wrongly.**
`date_parallel_lean.py:105`:

```python
pos = df_asset_rets.index.searchsorted(rebalance_time_index, side="right") - 1
```

A rebalance timestamp earlier than every panel row gives `searchsorted -> 0`, so
`pos = -1`. Two separate things then go wrong, and the mechanism is **not** the
lookahead the first draft of this plan described:

* The gates are read from the **last** panel row: `date_ok[-1]` and
  `cs_valid[-1]`. So whether the date is skipped at all depends on an unrelated
  date's fittability.
* The window is **empty**, not the full history: `n_full = i + 1 = 0`, so
  `a[:0]` and `valid[:0]` are `(0, N)`. Meanwhile `window_weights(pw, 0)` returns
  `pw[-0:]`, which is the **whole** array -- `-0 == 0`, so the negative-slice idiom
  silently degrades to "everything". numpy permits a length-0 boolean index against
  it (a length-2 one raises `IndexError`; length-0 returns `[]`), so nothing
  complains.

Measured by transcribing `_fit_core` to numpy and running it at `i = -1`:

```
n_full = 0 ; cols admitted: [0 1 2] ; a_win shape: (0, 3)
params written: [[0. 0. 0.]]
```

`pinv` of a `(0, K+1)` design returns a zero coefficient vector, so the date is
written **beta = 0.0**. The numba `_fit_core` does the same rather than raising --
`fit_huber_rlm_params(np.empty(0), np.empty((0, 2)))` returns `[0. 0.]`, measured,
so there is no crash to fall back on. `sim()` produces neither: the mask is empty,
`all(obj.empty)` is true, the callback is skipped and the date stays NaN.

**Impact is latent, not corrupting.** A rebalance timestamp before the panel is by
definition absent from `df_asset_rets.index`, so its buffer row is dropped by the
reindex onto the panel timestamps before the ffill ever runs. Verified: with a
pre-panel rebalance timestamp, all four returned frames match `parallel=False`
exactly, and the timestamp does not appear in `df_betas`. So this fabricates a
window, fits it and writes a wrong row, and then throws the row away.

Fix it anyway, with two guards rather than one: reject `pos < 0` in `precompute`
so no worker is dispatched a fabricated window, and reject `n_full <= 0` in
`fit_one_date`. `window_weights(pw, 0)` silently returning the whole panel is the
underlying trap and must be fixed at the source -- the next caller may not have a
reindex downstream to absorb it.

**5.2 The near-zero-variance guard both degrades and raises, inconsistently.**
Resolved as: **the date path replicates the serial path exactly**, both in which
behaviour fires and in how the variance is computed.

The serial path has two distinct checks, in this order inside
`_validate_beta_inputs`:

| | measured over | outcome | source |
|---|---|---|---|
| no-asset-columns early return | -- | all-NaN, no raise | `robust_betas.py:227` |
| **Check A** | each factor's own finite values in the window | **raise `ValueError`** | `robust_betas.py:237` |
| min-timestamps gate | complete rows | all-NaN | `robust_betas.py:267` |
| **Check B** | rows where every factor is finite | all-NaN (degrade) | `robust_betas.py:290` |

Check A runs **before** the min-timestamps gate and before Check B. The prototype
runs its Check A only over `date_ok` dates -- and `date_ok` is exactly what Check
B and the min-timestamps gate have already turned off. The two populations differ,
so the flat dates never reach the raise:

> A factor is constant for the panel's first 200 rows, then starts moving. Serial
> raises at date 50. The prototype skips date 50 via Check B, never runs Check A
> on it, and by date 300 the factor is moving so Check A passes. Same input: one
> raises, one returns quietly-NaN early dates.

Required behaviour: run Check A over the same date population serial does --
every rebalance date where at least one asset has listed **and** all factors are
live -- not over `date_ok`. Order is irrelevant (the message carries no
timestamp), so "raise if any candidate date fails" is exact.

**Second half, same item: the variance must be computed the same way.** The
prototype computes Check B one-pass from cumulative sums,
`(cs_fc2[i] - cs_fc[i]**2 / n) / (n - 1)`; serial uses numpy's two-pass
`.var(ddof=1)`. Algebraically identical, not identical in floating point: the
one-pass form subtracts two large nearly-equal numbers, so the leading digits
cancel and what survives is rounding noise of order `eps * sum_of_squares`. For
return-scale data (~1e-2) that is ~1e-17 against a 1e-10 threshold -- seven orders
of margin. For a factor whose values sit far from zero (a level near 100 rather
than a return near 0.01) it grows to ~1e-9, **larger** than the threshold, and a
date serial calls "flat, skip" the date path calls "fine, fit it". No error, just
different betas.

So drop `cs_fc` / `cs_fc2` entirely:

* Check A: call `finite_column_variances(f[:i+1])` -- the same function serial
  calls, on the same values.
* Check B: take `fc = f[complete_rows]` once, then score date `i` on
  `fc[:cs_complete[i]].var(axis=0, ddof=1)`. That slice holds exactly the rows
  `x_fact[complete_rows]` holds in the serial path, in the same order, contiguous
  and float64, so the reduction is bit-identical.

Cost is O(T^2 K) in the parent -- ~2M element operations at T=1393, K=1, against a
~6000 s fit stage. Not a consideration.

**5.3 The column-admission rule is re-derived, not shared.** The asset path drops
all-NaN columns in `sim()` and then requires `count_nonzero(mask) >= min_timestamps`
in `_fit_asset_chunk`; the date path uses `cs_valid[i] >= min_timestamps`.

Verified to agree today: `isfinite(y * sqrt_w) == isfinite(y)` for any finite
`sqrt_w` in `(0, 1]` (a zero weight maps a finite `y` to a finite `0` and an
infinite `y` to `NaN`, and both sides agree on each), and `sim()`'s all-NaN column
drop is subsumed by the `>= min_timestamps` count. Nothing enforces it. Keep the
derivation in one function whose docstring quotes the `sim()` rule it replaces.

**5.4 Panel spill location.** `betas_date_parallel` uses `tempfile.mkdtemp`, i.e.
`$TMPDIR`. The original handoff proposed preferring `/dev/shm`. **Do not.** We
have a measured ENOSPC failure from spooling betas data to `/dev/shm`.

But "keep `$TMPDIR`" is not the fix either: if `$TMPDIR` is tmpfs on the deploy
host, the spill *is* RAM and the ENOSPC risk returns with it. Require a
**disk-backed** directory: check the target filesystem is not tmpfs, check free
space before writing, and keep the `shutil.rmtree` in the `finally` that is
already there.

The spill is **~380 MB** at production width, not 234: `a` 234 MB + `valid`
(T, N) bool 29 MB + `cs_valid` (T, N) int32 117 MB.

**5.5 `fill_missing_betas` fills columns the serial path never touches.** This is
a confirmed divergence, not a risk to check.

Serial fills inside the per-date callback, on `beta_matrix`, whose columns are
only the assets `sim()` passed -- the assets live at that date. An asset that has
not listed yet is absent, so it is never written and stays NaN. The prototype
fills the whole buffer after the loop (`date_parallel_lean.py:224-228`), so an
unlisted asset gets **beta = 1.0**.
`tests/stats/test_pit_robust_betas_parallel_equiv.py:185` asserts exactly the
opposite ("B has no data and must stay NaN"), so the prototype fails an existing
test.

Two further mismatches in the same three lines: serial's `notna()` / `fillna(1.0)`
are NaN-based, the prototype's `isfinite` is not (they differ on `+/-inf` betas).

Correct form, in the parent after the buffer is filled: for each date row where
any beta is non-NaN, replace NaN with 1.0 **restricted to the assets live at that
date** (`first_notna_row[j] <= pos[d]`, derived from `~isnan`, matching `sim()`'s
`_first_valid_dates`, which uses `notna`). Liveness is a length-N comparison per
date, so no `(T, N)` array is needed.

Note the per-factor-row `row_has_any` is uniform across a date's K rows: an
admitted asset receives all K betas at once, so either every factor row of that
date has a non-NaN or none does.

## 6. Equivalence gates (Phase 3) -- the real gate, not a nicety

`tests/stats/test_pit_robust_betas_date_axis_equiv.py`, modelled on the existing
`test_pit_robust_betas_parallel_equiv.py`. Assert bit-identity of **all four**
returned frames against `parallel=False`, on both the clean and the hostile panel
(`make_hard_panel`: scattered non-finite factor rows, a late-listing factor, a
dead asset, a near-dead asset).

The prototype's own `bench.py --check` is not this gate and must not be read as
one. It compares only `df_betas` and `df_asset_resids`, at default arguments, and
never sets `fill_missing_betas`, never passes a `rebalance_time_index`, never
builds a flat-prefix factor and never touches `df_hedge_rets*`. None of 5.1, 5.2
or 5.5 is inside its coverage, so "bit-identical on clean and hard panels" is true
and much narrower than it sounds.

Cases that must be covered, each chosen because it is where the two paths derive
the same rule differently:

| case | what it pins |
|---|---|
| rebalance timestamp before the panel starts | 5.1 -- a regression test, not a reproduction: the bad row is reindexed away, so this pins that it stays that way |
| rebalance timestamps absent from the panel index | `searchsorted` vs `_mask_to` |
| `rebalance_time_index` a strict subset | row addressing in the buffer |
| single asset / single factor | degenerate shapes |
| a factor that lists late | `all_facts_live_from` vs the `n_facts` gate |
| an asset with exactly `min_timestamps - 1` and exactly `min_timestamps` rows | 5.3 boundary |
| a factor flat over a prefix of the panel | 5.2 raise-vs-skip, both paths must raise |
| `fill_missing_betas=True` with an unlisted asset | 5.5 |
| duplicate names / duplicate timestamps | the guards still fire before either path |
| `K > 1` | cross-factor row ordering in `betas_buf` |
| `n_jobs=1` | the sequential-backend initializer gap (3.2) |

## 7. Memory (Phase 4) -- deferred, not closed

The date axis measured **2794 MB peak PSS against 2519 MB**. Per the decision in
§0 this is **not** a gate on the default; it is deferred. Recorded here so
whoever picks it up does not restart the analysis.

Candidates, in order of suspicion:

1. **The panel is held twice.** `precompute` takes
   `np.ascontiguousarray(df_asset_rets.to_numpy())` while the caller's frame is
   still alive, then spills that copy and re-maps it. Peak inside `precompute`,
   before the pool even starts: caller frame 234 + `a` 234 + `cs_valid` 117 +
   `valid` 29 ~= 614 MB. This alone is the right order for the 275 MB gap.
2. `valid` (29 MB) and `cs_valid` (117 MB) are full-size arrays the asset path
   never builds at all.
3. In-flight block results, sized by `blocks_per_worker` (4.4).

**Not** a candidate: the `(n_dates * n_facts, n_assets)` betas buffer. It is 234 MB
and it is allocated identically on both paths, so it cannot explain a difference.
The earlier draft of this plan led with it; that was wrong.

Also note the asset path's per-date transient (2.09x the live window slice, 488 MB
at the widest window) is allocated *and freed* every date. It overlaps the betas
buffer at peak but does not ratchet.

When this is picked up: add a regression test asserting the date path's peak tree
PSS is not worse than the asset path's on a small panel, and re-measure at
production width with `bench.py --sizes 1393x21000 --K 1 --jobs 6`.

## 8. Rollout (Phase 5)

1. Land Phases 1-3 with `parallel_axis="asset"` as the default. Everything is
   dead code until someone opts in.
2. Gates green: the equivalence suite of §6.
3. Flip the default to `"date"` in its **own** commit, so it is revertible without
   touching the implementation. The PR body states the accepted +275 MB.
4. Keep `"asset"` reachable and tested indefinitely -- it is a supported choice,
   not a deprecation path.

## 9. `dagster-pipelines-internal` follow-ups

* `calculate_betas.py` needs no change to adopt the new default, which is exactly
  why the axis must be overridable without a release. Thread `parallel_axis` from
  `BETAS_PARALLEL_AXIS` alongside the existing `BETAS_*` knobs **before** the
  default flips, so a production incident has a fallback.
* Revisit `BETAS_PARALLEL_JOBS`. Per-worker memory on the date axis is a flat
  ~77 MB and the panel is shared, so the cap may be raisable -- but peak PSS is
  already the worse of the two axes, so re-measure rather than assume.

## 10. Prerequisites already landed

Both are in this repo and independent of the axis work:

* `Parallel(..., max_nbytes=None)` on the date-loop pool
  (`pit_robust_betas.py:373`) -- fixes a ~163 GB `/dev/shm` spool that failed a
  full-width build with ENOSPC, at +3.1% wall clock.
* `return_hedge_rets_by_fact` (`pit_robust_betas.py:459`) -- frees the largest
  returned frame early. Saves 234 MB held; **0 MB at peak**, contrary to the
  original handoff's claim.

Neither has reached `main`, which is **27** commits behind `dev-matt`. Nothing in
this plan deploys until that is resolved.

## 11. Reproducing the measurements

```bash
cd dagster-pipelines-internal/scratch/pit_betas_bench

python3 bench.py --check                                    # equivalence (narrow, see §6)
python3 bench.py --sizes 1393x21000 --K 1 --jobs 6 \
        --variants asset_par date_par_lean                  # production shape
python3 probe_perdate.py fit --T 1393 --N 21000 --K 1 --jobs 6
python3 probe_mem.py workers date_par_lean                  # per-worker Pss
```

`bench_common.py:REPO` must point at the checkout under test. Panels are
synthetic; real ones are raggeder, and raggedness changes load balance on both
axes.
