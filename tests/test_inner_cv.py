
"""Inspect nested temporal CV inner folds."""

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv
from src.tuning_validation import make_inner_folds


def main():
    config = load_config()
    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    outer_folds = temporal_cv(
        train,
        config["validation"]["cv_years"],
        config["validation"]["target_horizon_days"],
    )

    for year, outer_train, outer_val in outer_folds:
        inner_folds = make_inner_folds(outer_train, config)

        print(f"\n=== OUTER {year} ===")
        print(f"Outer train: {len(outer_train)}")
        print(f"Outer validation: {len(outer_val)}")
        print(f"Inner folds: {len(inner_folds)}")

        for i, fold in enumerate(inner_folds, start=1):
            inner_train = fold["train"]
            inner_val = fold["val"]

            print(
                f"  Inner {i}: "
                f"train={len(inner_train)}, "
                f"val={len(inner_val)}, "
                f"start={fold['validation_start'].date()}, "
                f"end={fold['validation_end'].date()}"
            )

            assert (
                inner_train["purchase_date"].max()
                + __import__("pandas").Timedelta(
                    days=config["validation"]["target_horizon_days"]
                )
                < fold["validation_start"]
            )

            assert (
                inner_val["purchase_date"].max()
                < outer_val["purchase_date"].min()
            )


if __name__ == "__main__":
    main()
