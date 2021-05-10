"""
Creates a dataframe of random hyperparameters as defined by the space in constants.py
"""
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
import pandas as pd

import setup as cons

np.random.seed(149817)


def sample_hps(hp_space: Dict[str, List[Union[int, str]]]) -> Dict[str, Union[int, str]]:
    sample = {}
    for k, v in hp_space.items():
        if len(v) == 1:
            sample[k] = v[0]
        elif isinstance(v[0], float):
            sample[k] = np.random.uniform(v[0], v[1])
        elif isinstance(v[0], int):
            sample[k] = np.random.choice(np.arange(v[0], v[1]+1))
        elif isinstance(v[0], str):
            sample[k] = np.random.choice(v)
    return sample


def build_hp_sample() -> pd.DataFrame:
    hps = {k: [] for k in cons.HP_SPACE.keys()}

    for i in range(cons.NUM_TRIALS):
        tmp = sample_hps(cons.HP_SPACE)
        for k in hps.keys():
            hps[k].append(tmp[k])

    df = pd.DataFrame.from_dict(hps)
    return df


def save_sample(df: pd.DataFrame) -> None:
    out_dir = Path('./')
    df.to_hdf(out_dir.joinpath('hp_sample.h5'), key='hyperparameters')


def main() -> None:
    hp_df = build_hp_sample()
    save_sample(hp_df)


if __name__ == '__main__':
    main()