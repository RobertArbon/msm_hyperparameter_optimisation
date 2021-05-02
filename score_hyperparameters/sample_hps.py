import numpy as np
import pandas as pd

import constants as cons

np.random.seed(149817)


def sample_hps(hp_space):
    sample = {}
    for k, v in hp_space.items():
        if isinstance(v[0], float):
            sample[k] = np.random.uniform(v[0], v[1])
        if isinstance(v[0], int):
            sample[k] = np.random.choice(np.arange(v[0], v[1]+1))
        if isinstance(v[0], str):
            sample[k] = np.random.choice(v)
    return sample


if __name__ == '__main__':
    hps = {k: [] for k in cons.HP_SPACE.keys()}
    for i in range(cons.NUM_HPS):
        tmp = sample_hps(cons.HP_SPACE)
        for k in hps.keys():
            hps[k].append(tmp[k])

    df = pd.DataFrame.from_dict(hps)
    df.to_hdf(f'{cons.NAME}.h5', key='hyperparameter_samples', mode='a')