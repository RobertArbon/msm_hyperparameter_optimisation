import pyemma as pm
import mdtraj as md
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, Union, List
from msmsense.featurizers import dihedrals, distances
from msmsense.bootstrap_cmatrices import get_sub_dict, get_trajs
from functools import partial
import functions as funcs
import sys


import pickle
import time
import seaborn as sns



def get_feature_dict(df, row_num):
    row_dict = df.filter(regex='__', axis=1).iloc[row_num, :].to_dict()
    feature_dict = get_sub_dict(row_dict, 'feature')
    if feature_dict['value'] == 'distances':
        feature_dict.update(get_sub_dict(row_dict, 'distances'))
    if feature_dict['value'] == 'dihedrals':
        feature_dict.update(get_sub_dict(row_dict, 'dihedrals'))
    return feature_dict

def get_kws_dict(df, row_num, kws):
    row_dict = df.filter(regex='__', axis=1).iloc[row_num, :].to_dict()   
    kws_dict = get_sub_dict(row_dict, kws)
    return kws_dict

def set_proper_dtypes(df):
    """
    forgot to save integers as integers. Only the distances feature columns have true floats. 
    """
    float_cols = list(df.filter(regex='distances.*', axis=1)) + list(df.filter(regex='.*vamp.*', axis=1)) 
    float_cols = float_cols + list(df.filter(regex='.*gap.*', axis=1)) 
    potential_integer_cols = df.columns.difference(float_cols)
    for col in potential_integer_cols:
        if str(df[col].dtype) != 'object':
            df[col] = df[col].astype(int)
    return df

def get_trajs_top(traj_dir: Path, protein_dir: str, rng: Union[np.random.Generator, None]=None):
    trajs = list(traj_dir.rglob(f"*{protein_dir.upper()}*/*.xtc"))
    trajs.sort()
    if rng is not None:
        ix = rng.choice(np.arange(len(trajs)), size=len(trajs), replace=True)
        trajs = [trajs[i] for i in ix]
    
    top = list(traj_dir.rglob(f"*{protein_dir.upper()}*/*.pdb"))[0]
    
    return {'trajs': trajs, 'top': top}
    
    
def get_random_traj(trajs: List[md.Trajectory], num_frames: int, rng: np.random.Generator)-> md.Trajectory: 
    traj_ix = np.arange(len(trajs))
    frame_ix = [np.arange(traj.n_frames) for traj in trajs]
    
    rand_ix = [(ix, rng.choice(frame_ix[ix])) for ix in rng.choice(traj_ix, size=num_frames)]
    rand_traj = md.join([trajs[x[0]][x[1]] for x in rand_ix])
    return rand_traj
    


# In[3]:


class MSM(object):
    
    def __init__(self, lag: int, num_evs: int, trajs: List[md.Trajectory], top: md.Trajectory,
                 feature_kws: Dict[str, Union[str, int, float]], tica_kws: Dict[str, Union[str, int, float]], cluster_kws: Dict[str, Union[str, int, float]], seed: int):
        """
        Defines the whole MSM pipeline.
        lag: markov lag time 
        num_evs: number of eigenvectors in VAMP score. This includes stationary distribution. note: all projections are done onto the processes 1 - num_evs, i.e., exclude the stationary distribution (process 0)
        traj_top_paths: dictionary with 'trajs' - list of Paths to trajectories, and 'top' Path to topology file. 
        
        """
        self.lag = lag
        self.num_evs = num_evs
        self.trajs = trajs

        self.top = top
        self.feature_kws = feature_kws
        self.tica_kws = tica_kws
        self.cluster_kws = cluster_kws
        self.featurizer = None
        self._set_featurizer()
        self.seed = seed

        self.ttrajs = None
        self.tica = None
        self.cluster = None
        self.msm = None
        self.paths = None
        
    def _set_featurizer(self):
        feature_kws = self.feature_kws.copy()
        feature = feature_kws.pop('value')
        
        if feature == 'distances':
            self.featurizer = partial(distances, **feature_kws)
        elif feature == 'dihedrals':
            self.featurizer = partial(dihedrals, **feature_kws)
        else:
            raise NotImplementedError('Unrecognized feature')
        

    def fit(self):
        ftrajs = self.featurizer(self.trajs)
        self.tica = pm.coordinates.tica(data=ftrajs, **self.tica_kws)
        ttrajs = self.tica.get_output()
        self.ttrajs = ttrajs
        self.cluster = pm.coordinates.cluster_kmeans(data=ttrajs, **self.cluster_kws, fixed_seed=self.seed)
        dtrajs = self.cluster.dtrajs
        self.msm = pm.msm.estimate_markov_model(dtrajs=dtrajs, lag=self.lag)

    
    def _get_all_projections(self, num_procs: int, dtrajs: List[np.ndarray]) -> np.ndarray:
        """ Project dtrajs onto first num_proc eigenvectors excluding stationary distribution. i.e., if num_proc=1 then project onto the slowest eigenvector only. 
        All projections ignore the stationary distribution
        """
        evs = self.msm.eigenvectors_right(num_procs+1)
        active_set = self.msm.active_set
        NON_ACTIVE_PROJ_VAL = 0 # if the state is not in the active set, set the projection to this value. 
        NON_ACTIVE_IX_VAL = -1
        evs = evs[:, 1:] # remove the stationary distribution
        proj_trajs = []
        for dtraj in dtrajs:
            all_procs = []
            for proc_num in range(num_procs):
                
                tmp = np.ones(dtraj.shape[0], dtype=float)
                tmp[:] = NON_ACTIVE_PROJ_VAL
                
                for i in range(dtraj.shape[0]):
                    x = self.msm._full2active[dtraj[i]]
                    if x != NON_ACTIVE_IX_VAL:
                        tmp[i] = evs[x, proc_num]
                    tmp = tmp.reshape(-1, 1)
                
                all_procs.append(tmp)
            all_procs = np.concatenate(all_procs, axis=1)
            proj_trajs.append(all_procs)
        
        return proj_trajs
        
        
    def projection_paths(self, n_projs: Union[None, int]=None, proj_dim: Union[None, int]=None, n_points: int=100, n_geom_samples: int=100):
        """
        n_projs: number of paths to create. Default = None = num_evs - 1 (i.e., exclude stationary distribution)
        proj_dim: dimensionality of the space in which distances will be computed. Default = None = num_evs - 1 (i.e., exclude stationary distribution)
        n_points: number of points along the path (there may be less than this.)
        n_geom_samples: For each of the n_points along the projection path, n_geom_samples will be retrieved from the trajectory files. 
                        The higher this number, the smoother the minRMSD projection path. Also, the longer it takes for the path to be computed
        """
        
        if proj_dim is None:
            proj_dim = self.num_evs - 1
        if n_projs is None:
            n_projs = self.num_evs - 1
        
        projections = self._get_all_projections(num_procs=proj_dim, dtrajs=self.msm.discrete_trajectories_active)
        
        paths, _ = projection_paths(MD_trajectories=self.trajs, MD_top = self.top, projected_trajectories=projections, 
                                    n_projs=n_projs, proj_dim=proj_dim, n_geom_samples=n_geom_samples, n_points=n_points)
        self.paths = paths
        
        
    
    def get_projection_trajectory(self, proc_num: int, kind: str='min_rmsd') -> md.Trajectory:
        """
        Returns the projection trajectory for a specific dimension
        proc_num: the number process. Min value = 1
        """
        if proc_num == 0:
            raise ValueError("process_num must be >=1. Processes are indexed from 0 (stationary distribution). Process 1 is the slowest projected process.")
            
        return self.paths[proc_num-1][kind]['geom']
    
    
    def transform(self, new_trajectory: md.Trajectory) -> np.ndarray:
        """
        projects new trajectory onto the self.num_proc eigenvectors of the MSM
        """
        ftrajs = self.featurizer([new_trajectory])
        ttrajs = self.tica.transform(ftrajs)
        dtrajs = self.cluster.transform(ttrajs)
        projections = self._get_all_projections(num_procs=self.num_evs - 1, dtrajs=dtrajs)
        return projections
        
        


# # Detailed comparisons
# 
# This workbook fits the best hyperparameters for each feature and provides a more detailed comparison. This complements the model comparison in terms of eigenvector projections. 
# 

# In[4]:


def model_kwargs(mod_defs, row_num, trajs, top, seed):

    lag = int(mod_defs.chosen_lag.values[row_num])
    num_evs = int(mod_defs.new_num_its.values[row_num])
    feat_kws = get_feature_dict(mod_defs, row_num)
    tica_kws = get_kws_dict(mod_defs, row_num, 'tica')
    cluster_kws = get_kws_dict(mod_defs, row_num, 'cluster')

    kwargs = dict(lag = lag, num_evs=num_evs, trajs=trajs, top=top,  feature_kws=feat_kws, tica_kws=tica_kws, cluster_kws=cluster_kws, seed=seed)
    return kwargs


def fit_model(kwargs):
    model = MSM(**kwargs)
    model.fit()
    return model

def cktest(model, mlags=10):
    pass


def get_trajectories(traj_dir, protein_dir, rng):
    traj_paths = get_trajs_top(traj_dir, protein_dir, rng)
    traj_paths_str = dict(top=str(traj_paths['top']), trajs=[str(x) for x in traj_paths['trajs']])
    top = md.load(str(traj_paths['top']))
    trajs = [md.load(str(x), top=top) for x in traj_paths['trajs']]
    return trajs, top
             


def get_model_defs(all_models, protein, feature):
    mod_defs = all_models.loc[all_models.protein==protein, :].copy()
    row_num = np.where(mod_defs['feature'].values==feature)[0][0]
    return mod_defs, row_num


def bootstrap_cktest(traj_dir, protein_dir, rng, mod_defs, row_num, num_bootstraps=100):
    
    cktests = {'predictions': [], 'estimates': []}
    
    for i in range(num_bootstraps):
        print('\t', i)
        trajs, top = get_trajectories(traj_dir, protein_dir, rng)
        kwargs = model_kwargs(mod_defs, row_num, trajs, top, seed)
        model = fit_model(kwargs)
        cktest = model.msm.cktest(nsets=model.num_evs, mlags=10)
        cktests['predictions'].append(cktest.predictions[np.newaxis, ...])
        cktests['estimates'].append(cktest.estimates[np.newaxis, ...])
    
    cktests['predictions'] = np.concatenate(cktests['predictions'], axis=0)
    cktests['estimates'] = np.concatenate(cktests['estimates'], axis=0)
    return cktests

    


if __name__ == '__main__': 

    traj_dir = Path('/Volumes/REA/MD/12FF/strided/')
    m1_sel = set_proper_dtypes(pd.read_hdf('./summaries/m1_model_selection.h5'))
    m2_sel = set_proper_dtypes(pd.read_hdf('./summaries/m2_model_selection.h5'))
    model_selections={'m1': m1_sel, 'm2': m2_sel}
    
    prot_dict = dict(zip(funcs.PROTEIN_LABELS, funcs.PROTEIN_DIRS))
    
    num_bootstraps = 100 
    seed = 12098345
    
    features = m1_sel.feature.unique()
    
    
    rng = np.random.default_rng(seed)
    
    protein = sys.argv[1]
    
    # Setup output directory
    protein_dir = prot_dict[protein]
    root_dir = Path(f"ck_tests/{protein}")
    root_dir.mkdir(exist_ok=True)
    
    # loop over feature
    for feature in features: 
        
        # loop over selection methods
        for method, selection in model_selections.items(): 
            
            # Get model definition
            mod_defs, row_num = get_model_defs(selection, protein, feature)
    
            # create output path
            output_path = root_dir.joinpath(f"{method}_{feature.replace('.', '')}_hpix{mod_defs['hp_index'].values[row_num]}_cktest.p")
            
            print(output_path)
    
            # bootstrap cktest
            cktests = bootstrap_cktest(traj_dir, protein_dir, rng, mod_defs, row_num, num_bootstraps=num_bootstraps)
            pickle.dump(file=output_path.open('wb'), obj=cktests)
    
