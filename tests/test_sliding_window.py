"""Run with: python -m tests.test_sliding_window after copying into tests/."""
import pandas as pd
from src.validation import temporal_cv
from src.tuning_validation import make_inner_folds


def main():
    df = pd.DataFrame({
        'purchase_date': pd.date_range('2017-01-01', '2023-12-31', freq='D'),
    })
    years = [2020, 2021, 2022, 2023]
    expanding = list(temporal_cv(df, years, 90))
    sliding = list(temporal_cv(df, years, 90, 'sliding_window', 2))
    assert len(expanding) == len(sliding) == 4
    for (year, old_train, old_val), (_, train, val) in zip(expanding, sliding):
        assert old_val.index.equals(val.index)
        assert len(train) <= len(old_train)
        start = pd.Timestamp(f'{year}-01-01')
        assert train['purchase_date'].min() >= start - pd.DateOffset(years=2)
        assert train['purchase_date'].max() + pd.Timedelta(days=90) < start

    config = {
        'validation': {'target_horizon_days': 90, 'sliding_window': {'train_years': 2}},
        'hyperparameter_tuning': {'nested_cv': {'inner': {
            'strategy': 'sliding_window', 'max_splits': 3,
            'validation_window_days': 180, 'step_days': 180,
            'target_horizon_days': 90, 'min_train_rows': 100,
            'min_validation_rows': 100,
        }}},
    }
    inner = make_inner_folds(sliding[-1][1], config)
    assert 1 <= len(inner) <= 3
    for fold in inner:
        assert fold['train']['purchase_date'].min() >= fold['validation_start'] - pd.DateOffset(years=2)
        assert fold['train']['purchase_date'].max() + pd.Timedelta(days=90) < fold['validation_start']
    print('Sliding window tests passed. Inner folds:', len(inner))


if __name__ == '__main__':
    main()
