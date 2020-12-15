import pytest

from src.submit_jobs import all_combinations


def test_grid_search():
    assert(all_combinations([
        {
            'grid': {
                'normalise_with_max_returns': [False, True],
                'normalise_returns': [1],
                'center_returns': ['s'],
            }
        },
        {
            'grid': {
                'normalise_with_max_returns': [None],
                'normalise_returns': [2],
                'center_returns': ['a', 'b'],
            }
        },
    ]) == ([(False, 1, 's'), (True, 1, 's'), (None, 2, 'a'), (None, 2, 'b')], ['normalise_with_max_returns',
                                                                               'normalise_returns',
                                                                               'center_returns']))


def test_keys_wrong_order():
    with pytest.raises(AssertionError, match='KeyNotSame'):
        all_combinations([
            {
                'grid': {
                    'normalise_with_max_returns': [False, True],
                    'normalise_returns': [1],
                }
            },
            {
                'grid': {
                    'normalise_returns': [2],
                    'normalise_with_max_returns': [None],
                }
            },
        ])


def test_keys_not_same():
    with pytest.raises(AssertionError, match='KeyNotSame'):
        all_combinations([
            {
                'grid': {
                    'normalise_returns': [1],
                }
            },
            {
                'grid': {
                    'normalise_returns2': [2],
                }
            },
        ])


if __name__ == '__main__':
    test_grid_search()