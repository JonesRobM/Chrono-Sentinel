"""
Tests for data loading, labelling and splitting.

The split tests are regression guards. The original pipeline concatenated
every series and then split the concatenation temporally, which silently put
one whole series in train and another whole series in test. It looked like a
70/15/15 split and was in fact an out-of-distribution transfer test, and the
resulting model scored below chance. Nothing in the code said so, so these
tests say it instead.
"""

import numpy as np
import pandas as pd
import pytest

from threatsim.data import (
    DEFAULT_TEST_SERIES,
    DEFAULT_TRAIN_SERIES,
    DEFAULT_VAL_SERIES,
    FeatureScaler,
    create_windows,
    get_nab_root,
    nab_anomaly_mask,
    normalise_windows,
    prepare_grouped_splits,
)

nab_available = pytest.mark.skipif(
    not (get_nab_root() / "labels" / "combined_labels.json").exists(),
    reason="NAB data not present; run scripts/fetch_data.py",
)


def make_frame(n: int, freq_minutes: int = 5) -> pd.DataFrame:
    """Builds a synthetic series with a regular timestamp index."""
    start = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        {
            "timestamp": [
                start + pd.Timedelta(minutes=freq_minutes * i) for i in range(n)
            ],
            "value": np.arange(n, dtype=float),
        }
    )


class TestAnomalyMask:
    """nab_anomaly_mask implements NAB's own anomaly-window convention."""

    def test_no_anomalies_gives_empty_mask(self):
        mask = nab_anomaly_mask(make_frame(100), [], 0.1)
        assert mask.sum() == 0

    def test_window_width_follows_the_fraction(self):
        frame = make_frame(1000)
        anomaly = [frame["timestamp"].iloc[500].isoformat()]
        narrow = nab_anomaly_mask(frame, anomaly, 0.02)
        wide = nab_anomaly_mask(frame, anomaly, 0.10)
        assert wide.sum() > narrow.sum()
        # Total flagged length tracks the requested fraction, within rounding.
        assert narrow.sum() == pytest.approx(0.02 * 1000, abs=3)
        assert wide.sum() == pytest.approx(0.10 * 1000, abs=3)

    def test_budget_is_split_across_anomalies(self):
        """Two anomalies share the same total budget, not double it."""
        frame = make_frame(1000)
        one = nab_anomaly_mask(frame, [frame["timestamp"].iloc[300].isoformat()], 0.10)
        two = nab_anomaly_mask(
            frame,
            [
                frame["timestamp"].iloc[200].isoformat(),
                frame["timestamp"].iloc[800].isoformat(),
            ],
            0.10,
        )
        assert two.sum() == pytest.approx(one.sum(), abs=4)

    def test_mask_is_centred_on_the_annotation(self):
        frame = make_frame(1000)
        mask = nab_anomaly_mask(frame, [frame["timestamp"].iloc[500].isoformat()], 0.05)
        flagged = np.flatnonzero(mask)
        assert flagged.mean() == pytest.approx(500, abs=2)

    def test_duplicate_timestamps_do_not_raise(self):
        """Several NAB series contain duplicate timestamps."""
        frame = make_frame(100)
        frame.loc[50, "timestamp"] = frame.loc[49, "timestamp"]
        mask = nab_anomaly_mask(frame, [frame["timestamp"].iloc[10].isoformat()], 0.1)
        assert mask.sum() > 0


class TestWindowing:
    """create_windows and normalise_windows."""

    def test_shapes_and_stride(self):
        values = np.arange(100, dtype=np.float32)
        labels = np.zeros(100, dtype=np.int64)
        windows, window_labels = create_windows(
            values, labels, window_size=10, step_size=5
        )
        assert windows.shape == (19, 10)
        assert window_labels.shape == (19,)
        np.testing.assert_array_equal(windows[1], values[5:15])

    def test_window_is_positive_if_it_contains_any_anomaly(self):
        values = np.arange(100, dtype=np.float32)
        labels = np.zeros(100, dtype=np.int64)
        labels[12] = 1
        _, window_labels = create_windows(values, labels, window_size=10, step_size=5)
        assert window_labels[1] == 1  # covers 5:15
        assert window_labels[0] == 0  # covers 0:10

    def test_series_shorter_than_window_yields_nothing(self):
        windows, labels = create_windows(
            np.arange(5, dtype=np.float32), np.zeros(5, dtype=np.int64), 50, 10
        )
        assert windows.shape == (0, 50)
        assert labels.shape == (0,)

    def test_normalisation_is_per_window(self):
        windows = np.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]], dtype=np.float32)
        normalised = normalise_windows(windows)
        # Both rows collapse to the same shape: level and scale are discarded,
        # which is exactly why the model also needs the feature vector.
        np.testing.assert_allclose(normalised[0], normalised[1], rtol=1e-5)
        np.testing.assert_allclose(normalised.mean(axis=1), [0, 0], atol=1e-6)

    def test_constant_window_does_not_divide_by_zero(self):
        normalised = normalise_windows(np.full((1, 10), 7.0, dtype=np.float32))
        assert np.all(np.isfinite(normalised))


class TestFeatureScaler:
    """The scaler must round-trip exactly, since it is persisted and reloaded."""

    def test_transform_standardises(self):
        features = np.random.default_rng(0).normal(5, 3, (500, 10)).astype(np.float32)
        scaler = FeatureScaler.fit(features)
        scaled = scaler.transform(features)
        np.testing.assert_allclose(scaled.mean(axis=0), np.zeros(10), atol=1e-4)
        np.testing.assert_allclose(scaled.std(axis=0), np.ones(10), atol=1e-4)

    def test_round_trip_through_dict(self):
        features = np.random.default_rng(1).normal(0, 1, (100, 10)).astype(np.float32)
        scaler = FeatureScaler.fit(features)
        restored = FeatureScaler.from_dict(scaler.to_dict())
        np.testing.assert_allclose(
            scaler.transform(features), restored.transform(features), rtol=1e-5
        )

    def test_constant_feature_does_not_divide_by_zero(self):
        features = np.ones((50, 10), dtype=np.float32)
        scaled = FeatureScaler.fit(features).transform(features)
        assert np.all(np.isfinite(scaled))


@pytest.fixture(scope="module")
def splits():
    """Grouped splits, built once for the whole module."""
    return prepare_grouped_splits()


@nab_available
class TestGroupedSplits:
    """Regression guards on split integrity."""

    def test_no_series_appears_in_more_than_one_split(self):
        train = set(DEFAULT_TRAIN_SERIES)
        val = set(DEFAULT_VAL_SERIES)
        test = set(DEFAULT_TEST_SERIES)
        assert not train & val
        assert not train & test
        assert not val & test

    def test_overlapping_series_are_rejected(self):
        """A leaking configuration must fail loudly, not train quietly."""
        with pytest.raises(ValueError, match="leaks"):
            prepare_grouped_splits(
                train_series=DEFAULT_TRAIN_SERIES,
                val_series=DEFAULT_VAL_SERIES,
                test_series=[DEFAULT_TRAIN_SERIES[0]],
            )

    def test_every_split_contains_both_classes(self, splits):
        """A one-class split makes AUC and AP undefined."""
        data, _ = splits
        for name in ("train", "val", "test"):
            labels = data[name]["labels"]
            assert labels.sum() > 0, f"{name} has no positive windows"
            assert (labels == 0).sum() > 0, f"{name} has no negative windows"

    def test_positive_rates_are_comparable_across_splits(self, splits):
        """
        The original temporal split produced 0.6% positives in train and 4.8%
        in test. Rates that far apart mean the splits are not measuring the
        same problem.
        """
        data, _ = splits
        rates = [data[name]["labels"].mean() for name in ("train", "val", "test")]
        assert max(rates) / min(rates) < 3.0, f"positive rates diverge: {rates}"

    def test_feature_scaler_is_fitted_on_train_only(self, splits):
        """Train features are standardised; held-out features need not be."""
        data, _ = splits
        train_features = data["train"]["features"]
        np.testing.assert_allclose(
            train_features.mean(axis=0), np.zeros(train_features.shape[1]), atol=1e-3
        )

    def test_windows_are_normalised(self, splits):
        data, _ = splits
        for name in ("train", "val", "test"):
            windows = data[name]["windows"]
            np.testing.assert_allclose(
                windows.mean(axis=1), np.zeros(len(windows)), atol=1e-4
            )
