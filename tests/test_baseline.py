import numpy as np

from src.models.baseline import held_out_files_split, time_contiguous_split


def test_time_contiguous_split_holds_out_end_of_each_file():
    # file 0 has 10 windows, file 1 has 5 windows
    file_ids = np.array([0] * 10 + [1] * 5)
    train_mask, test_mask = time_contiguous_split(file_ids, test_fraction=0.2)

    assert not np.any(train_mask & test_mask)
    assert np.all(train_mask | test_mask)
    # last 2 windows of file 0 (indices 8,9) held out
    assert test_mask[8] and test_mask[9]
    assert not test_mask[7]
    # last 1 window of file 1 (index 14) held out
    assert test_mask[14]
    assert not test_mask[13]


def test_held_out_files_split():
    file_ids = np.array([0, 0, 1, 1, 2, 2])
    train_mask, test_mask = held_out_files_split(file_ids, test_file_ids={1})

    assert np.array_equal(test_mask, np.array([False, False, True, True, False, False]))
    assert np.array_equal(train_mask, ~test_mask)
