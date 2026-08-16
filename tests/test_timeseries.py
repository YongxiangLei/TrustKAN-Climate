import numpy as np
import pytest

from src.data.timeseries import chronological_split, sliding_windows, TrainOnlyStandardizer, assign_windows_by_target_origin


def test_chronological_split_is_contiguous():
    s = chronological_split(100, train=0.6, val=0.15, calibration=0.1)
    assert s.train == slice(0, 60)
    assert s.val == slice(60, 75)
    assert s.calibration == slice(75, 85)
    assert s.test == slice(85, 100)


def test_standardizer_uses_training_statistics_only():
    train = np.array([0.0, 1.0, 2.0])
    future = np.array([100.0, 101.0])
    scaler = TrainOnlyStandardizer().fit(train)
    assert np.isclose(scaler.scaler.mean_[0], 1.0)
    transformed = scaler.transform(future)
    assert transformed.mean() > 50.0


def test_windows_and_origins_are_temporal():
    x, y, origins = sliding_windows(np.arange(10), history=3, horizon=2)
    assert np.array_equal(x[0, :, 0], [0, 1, 2])
    assert np.array_equal(y[0], [3, 4])
    assert origins[0] == 3


def test_window_assignment_by_target_origin():
    split = chronological_split(20, train=0.5, val=0.2, calibration=0.1)
    _, _, origins = sliding_windows(np.arange(20), history=3, horizon=1)
    masks = assign_windows_by_target_origin(origins, split)
    assert np.all(origins[masks["train"]] < split.train.stop)
    assert np.all(origins[masks["test"]] >= split.test.start)


def test_multi_step_targets_do_not_cross_split_boundaries():
    split = chronological_split(20, train=0.5, val=0.2, calibration=0.1)
    _, _, origins = sliding_windows(np.arange(20), history=3, horizon=3)
    masks = assign_windows_by_target_origin(origins, split, horizon=3)
    for name, region in (
        ("train", split.train),
        ("val", split.val),
        ("calibration", split.calibration),
        ("test", split.test),
    ):
        selected = origins[masks[name]]
        assert np.all(selected >= region.start)
        assert np.all(selected + 3 <= region.stop)

    assert not masks["train"][np.flatnonzero(origins == split.train.stop - 1)[0]]


def test_split_rejects_empty_segments_after_rounding():
    with pytest.raises(ValueError, match="empty chronological segment"):
        chronological_split(8, train=0.6, val=0.15, calibration=0.1)
