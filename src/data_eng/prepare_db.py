import pandas as pd
import numpy as np
from src.config.configuration import DATA_DIR
import os


def load_dataset(path_to_file: str) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(DATA_DIR, path_to_file), sep=";")
    return data


def split_dataset(input_df: pd.DataFrame, train_pct: float) -> [pd.DataFrame, pd.DataFrame]:
    df_len = len(input_df)
    input_df = input_df.sample(frac=1).reset_index(drop=True)
    train, test = np.split(input_df, [int(train_pct * df_len)])
    return train, test


def check_variable_is_monotonic(values: pd.Series) -> bool:
    return len(values.unique()) == 1
