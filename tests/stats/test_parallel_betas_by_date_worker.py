"""Lifecycle tests for the date-axis worker.

The date-axis equivalence suites assert the numbers a run produces, and both
would keep passing if the worker opened its arrays eagerly again: the eager
version returned correct betas and then logged ``FileNotFoundError`` at CRITICAL
from every worker that started after the parent had finished. What these tests
pin is therefore the lifecycle rather than the arithmetic -- when the mappings
are opened, that they are opened once, and that a worker which never receives a
block never touches the files at all.

The worker is driven directly instead of through ``joblib``. The failure being
guarded against is a race between worker startup and the parent's cleanup, and a
pool cannot be made to lose that race on demand; calling the two halves in the
order the race produces reproduces it exactly and takes no pool to do it.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vbase_utils.stats import _parallel_betas_by_date_worker as worker
from vbase_utils.stats._parallel_betas_by_date import (
    META_KEYS,
    SPILL_KEYS,
    precompute_date_betas,
)
from vbase_utils.stats._parallel_betas_by_date_worker import (
    _G,
    fit_date_group,
    initialize_date_worker,
)

from ._robust_betas_fixtures import make_linear_ret_frames as _make_returns

# Panel size. Long enough that late dates clear MIN_TIMESTAMPS and early ones do
# not, which is what gives the tests both a fitted and a rejected date.
N_TIMESTAMPS = 60
MIN_TIMESTAMPS = 20
HALF_LIFE = 30.0
SEED = 11

# Rows used to build the groups sent to the worker. The first is rejected for
# having fewer than MIN_TIMESTAMPS complete rows behind it; the last has the
# whole panel behind it and is fit.
REJECTED_ROW = 0
FITTED_ROW = N_TIMESTAMPS - 1


def _make_precomputed() -> dict:
    """Return the precomputed inputs for a small two-asset, one-factor panel."""
    df_asset, df_fact = _make_returns(
        n=N_TIMESTAMPS,
        betas={"A": [0.7], "B": [1.3]},
        factors=["F1"],
        seed=SEED,
    )
    return precompute_date_betas(
        df_asset_rets=df_asset,
        df_fact_rets=df_fact,
        rebalance_time_index=pd.DatetimeIndex(df_asset.index),
        half_life=HALF_LIFE,
        lambda_=None,
        min_timestamps=MIN_TIMESTAMPS,
    )


class TestDateWorkerLifecycle(unittest.TestCase):
    """The worker records paths at startup and opens the arrays on first use."""

    def setUp(self):
        pc = _make_precomputed()
        self.tmpdir = tempfile.mkdtemp(prefix="vbase_date_worker_test_")
        # Spill exactly what the parallel path spills, through the same key list,
        # so this test cannot drift from the producer.
        self.paths = {}
        for key in SPILL_KEYS:
            path = os.path.join(self.tmpdir, f"{key}.npy")
            np.save(path, pc[key])
            self.paths[key] = path
        self.meta = {k: pc[k] for k in META_KEYS}
        self.asset_names = ["A", "B"]

    def tearDown(self):
        # The worker's state is a module global, so a test that leaves a mapping
        # behind would hand it to the next one.
        _G.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initializer_does_not_open_the_arrays(self):
        """Startup records the paths and loads nothing."""
        with patch.object(worker.np, "load") as mock_load:
            initialize_date_worker(self.paths, self.meta, self.asset_names)
            mock_load.assert_not_called()
        self.assertEqual(_G["paths"], self.paths)
        self.assertNotIn("pc", _G)

    def test_initializer_survives_the_spill_directory_being_gone(self):
        """A worker that starts after the parent has cleaned up still starts.

        This is the race the lazy opening was introduced for: loky does not wait
        for the initializer to finish, so a worker could still be starting when
        the parent had collected every block and removed the spill directory. A
        worker in that position has no block left to fit, so recording the paths
        must not depend on the files still being there.
        """
        shutil.rmtree(self.tmpdir)
        initialize_date_worker(self.paths, self.meta, self.asset_names)
        self.assertNotIn("pc", _G)

    def test_first_group_opens_the_arrays_and_later_groups_reuse_them(self):
        """The mappings are opened by the first task and then held."""
        initialize_date_worker(self.paths, self.meta, self.asset_names)
        group = [(0, FITTED_ROW)]

        fit_date_group(group)
        opened = _G["pc"]
        self.assertIsNotNone(opened)

        with patch.object(worker.np, "load") as mock_load:
            fit_date_group(group)
            mock_load.assert_not_called()
        self.assertIs(_G["pc"], opened)

    def test_a_second_initialization_drops_the_previous_mapping(self):
        """Re-initializing clears the mapping so it cannot outlive its paths."""
        initialize_date_worker(self.paths, self.meta, self.asset_names)
        fit_date_group([(0, FITTED_ROW)])
        self.assertIn("pc", _G)

        initialize_date_worker(self.paths, self.meta, self.asset_names)
        self.assertNotIn("pc", _G)


class TestDateGroupDateCount(unittest.TestCase):
    """A group reports how many dates it covered, not how many it fit.

    The parent's progress postfix counts dates through this number. Taking it
    from the returned fits instead undercounts every group holding a rejected
    date, and reports no progress at all for a run whose dates are all rejected
    -- which is a real outcome, not a hypothetical one: a build whose panel is
    too short to clear ``min_timestamps`` fits nothing and must still show its
    blocks completing.
    """

    def setUp(self):
        pc = _make_precomputed()
        self.tmpdir = tempfile.mkdtemp(prefix="vbase_date_worker_test_")
        paths = {}
        for key in SPILL_KEYS:
            path = os.path.join(self.tmpdir, f"{key}.npy")
            np.save(path, pc[key])
            paths[key] = path
        initialize_date_worker(paths, {k: pc[k] for k in META_KEYS}, ["A", "B"])

    def tearDown(self):
        _G.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mixed_group_counts_every_date(self):
        """One fitted and one rejected date still count as two."""
        n_dates, fits = fit_date_group([(0, REJECTED_ROW), (1, FITTED_ROW)])
        self.assertEqual(n_dates, 2)
        self.assertEqual(len(fits), 1)

    def test_all_rejected_group_counts_its_dates(self):
        """A group that fits nothing reports the dates it covered."""
        n_dates, fits = fit_date_group([(0, REJECTED_ROW), (1, REJECTED_ROW)])
        self.assertEqual(n_dates, 2)
        self.assertEqual(fits, [])


if __name__ == "__main__":
    unittest.main()
