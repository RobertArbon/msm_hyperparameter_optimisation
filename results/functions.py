import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import plotly.express as px
from typing import List
import re

cols = sns.color_palette('colorblind')

PROTEIN_DIRS = ['1fme', '2f4k', '2jof', '2wav', 'cln025', 'gtt', 'prb', 'uvf', 'lambda', 'ntl9', 'nug2', 'a3d']
PROTEIN_LABELS = ['BBA', 'Villin', 'Trp-cage', 'BBL', 'Chignolin', 'WW-domain', 'Protein-B', 'Homeodomain', '$\lambda$-repressor', 'NTL9', 'Protein-G', r'$\alpha$3D']
LETTERS = list('abcdefghijklmnopqrstuvwxyz')

FIG_DIR =Path(__file__).absolute().parents[1].joinpath('figures')
assert FIG_DIR.exists(), "path to figures directory doesn't exist"



def get_df(hdf_path: Path, results: str) -> pd.DataFrame:
    ts = pd.read_hdf(hdf_path, key=results)
    return ts


def get_hp_index(paths: List[Path], parent: int=0) -> List[int]:
    index = [re.findall('[0-9]+', x.parents[parent].stem)[0] for x in paths]
    index = [int(x) for x in index]
    return index


def get_results_df(results_paths, results: str):
    all_ts = []
    indices = get_hp_index(results_paths)
    for i, path in zip(indices, results_paths):
        ts = get_df(path, results)
        hp = pd.read_hdf(path, key='hp')
        hp['hp_index'] = i #path.parents[0].stem
        df = ts.join(hp).ffill()
        all_ts.append(df)
    all_ts = pd.concat(all_ts, axis=0)
    return all_ts


def timescale_gradient(ts_df: pd.DataFrame, x: str, log: bool = True, denom: str='one') -> pd.DataFrame:
    """
    Takes difference in timescales with respect to x.  Dataframe must be suitably subset before passing!
    """
    t = ts_df.loc[:, ['protein', x, 'value', 'hp_index', 'iteration']]
    
    dupes = t.duplicated(subset=['protein', x, 'hp_index', 'iteration'])
    
    assert not np.any(dupes.values), f'Duplicate values found: protein, {x}, index, & iteration, columns must be unique'
    
    if log:
        t['value'] = np.log(t['value'])
        
    t.sort_values(by=['protein', 'hp_index', 'iteration', x], inplace=True)
    t['delta_t'] = t.groupby(['protein', 'hp_index', 'iteration']).diff()['value']
    t['delta_x'] = t.groupby(['protein', 'hp_index', 'iteration']).diff()[x]
    
    if denom=='one':
        t['grad_t'] = t['delta_t']/1.0
    elif denom=='x':
        t['grad_t'] = t['delta_t']/t[x]
    elif denom=='delta_x':
        t['grad_t'] = t['delta_t']/t['delta_x']
        
    t.dropna(axis = 0, how = 'any', inplace = True)
    return t



def plot_timescales(ts, ax, label, errorbars=True):
#     lags = ts.lag.unique()
    nits = ts.num_its.unique()
    for its in nits:
        ax = plot_timescale(ts, its, ax, label, errorbars)
    return ax


def plot_timescale(ts, its, ax, label, errorbars=True):
    """
    ts = implied timescales df
    its = the implied timescale of interest
    """
    ix = ts.num_its == its
    x = ts.loc[ix, "lag"]
    y = ts.loc[ix, "median"]
    if errorbars:
        yerr = ts.loc[ix, ['del_lower', 'del_upper']].values.T
        ax.errorbar(x, y, yerr, lw=1, marker='o', elinewidth=2, color=cols[0])
    else:
        ax.plot(x, y, lw=1, label=label, color=cols[0])
    ax.set_yscale('log')
    return ax
    
    
def get_tau(tss, lags):
    """
    tss = implied timescales
    lags = associated markov lag time
    returned tau can't be the first lag
    """
    del_ts = tss[1:]/tss[:-1]
    min_grad = np.min(del_ts)
    tau = lags[1:][np.argmin(del_ts)]
    return tau, min_grad


def plot_tau(ts, its, ax, label):

    ix = ts.num_its == its
    lags = ts.loc[ix, "lag"].values
    tss = ts.loc[ix, "median"].values
    tau, grad = get_tau(tss, lags)
    ax.scatter(tau, grad, marker='o', s=50, alpha=1)
    return ax


def plot_taus(ts, ax, label):
    nits = ts.num_its.unique()
    for its in nits:
        ax = plot_tau(ts, its, ax, label)
    return ax