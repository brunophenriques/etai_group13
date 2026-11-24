"""Coordinate fitted preprocessing within temporal folds."""
from src.imputation import ImputationPreprocessor


def prepare_numeric_fold(train_fold, val_fold, config, task=None):
    """Keep the three-result interface; state can be reused at prediction."""
    preprocessor = ImputationPreprocessor(config, task).fit(train_fold)
    return (preprocessor.transform(train_fold), preprocessor.transform(val_fold),
            {"__preprocessor__": preprocessor})
