import pandas as pd
import numpy as np
from src.config.configuration import DATA_DIR
import os


def load_dataset(path_to_file: str) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(DATA_DIR, path_to_file), sep=";")
    return data


def split_dataset(input_df: pd.DataFrame, train_pct: float, test_pct: float, valid_pct: float) -> [pd.DataFrame,
                                                                                                   pd.DataFrame,
                                                                                                   pd.DataFrame]:
    df_len = len(input_df)
    train, test, valid = np.split(input_df, [int(train_pct * df_len), int(test_pct * df_len)])
    return train, test, valid
