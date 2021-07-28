#!/usr/bin/env python
# coding: utf-8

# In[139]:


# import h5py as hp
import pickle
from msmtools.estimation import transition_matrix as _transition_matrix
from msmtools.analysis import timescales as _timescales
from pyemma.util.metrics import vamp_score
import numpy as np
from pathlib import Path
import pandas as pd
import sys

# This extracts timescales and vamp scores.
# In[6]:


def vamp(cmat, T, method, k):

    C0t = cmat
    C00 = np.diag(C0t.sum(axis=1))
    Ctt = np.diag(C0t.sum(axis=0))
    return vamp_score(T, C00, C0t, Ctt, C00, C0t, Ctt,
                          k=k, score=method)
    


# In[119]:

if __name__=='__main__':
	args = sys.argv
	if len(args) != 3:
		raise RuntimeError('score.py protein hp_ix')	
	data_dir = Path('/Volumes/REA/Data/fast_folders/')
	protein = args[1] #'1fme'
	hp_ix = args[2] 
	hp_dir = f"hp_{hp_ix}"
	
	bs_paths = list(data_dir.joinpath(protein, hp_dir).glob('*.pkl'))
	bs_results = [pickle.load(path.open('rb')) for path in bs_paths]
	print(f'Processing {len(bs_paths)} trials from {str(bs_paths[0].parent)}')
	
	vamp_ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	vamp_methods = ['VAMP1', 'VAMP2', 'VAMPE']
	
	num_iters = len(bs_results)
	num_lags = max([len(x['lags']) for x in bs_results])
	
	
	num_vamp_ks  = len(vamp_ks)
	num_vamp_methods = len(vamp_methods)
	num_its = 10
	
	timescales_values = np.zeros((num_iters, num_lags, num_its))
	lag_values = np.empty((num_iters, num_lags, num_its))
	its_values = np.empty((num_iters, num_lags, num_its))
	iter_values = np.empty((num_iters, num_lags, num_its))
	
	vamp_values = np.zeros((num_iters, num_lags, num_vamp_methods, num_vamp_ks))
	vamp_lag_values = np.empty((num_iters, num_lags, num_vamp_methods, num_vamp_ks))
	vamp_method_values = np.empty((num_iters, num_lags, num_vamp_methods, num_vamp_ks), dtype='object')
	vamp_ks_values = np.empty((num_iters, num_lags, num_vamp_methods, num_vamp_ks))
	vamp_iter_values = np.empty((num_iters, num_lags, num_vamp_methods, num_vamp_ks))
	
	for bs_idx in range(num_iters):
	    print(bs_idx, end=', ')
	    results = bs_results[bs_idx]
	    lags = results['lags']
	    for lag_idx in range(len(lags)):
	        lag = lags[lag_idx]
	        
	        cmat = results['count_matrices'][lag_idx]
	        T = _transition_matrix(cmat, reversible=True)
	        
	        # accumulated timescales
	        ts = _timescales(T, tau=lag)
	        n_its= min(num_its, ts.shape[0])
	        timescales_values[bs_idx, lag_idx][:n_its] = ts[1:n_its+1]        
	        
	        lag_values[bs_idx, lag_idx][:n_its] = lag
	        its_values[bs_idx, lag_idx][:n_its] = np.arange(n_its)+2
	        iter_values[bs_idx, lag_idx][:n_its] = bs_idx
	        
	        for meth_idx, method in enumerate(vamp_methods):
	            for k_idx, k in enumerate(vamp_ks):
	                vamp_values[bs_idx, lag_idx, meth_idx, k_idx] = vamp(cmat, T, method, k)
	                vamp_lag_values[bs_idx, lag_idx, meth_idx, k_idx] = lag
	                vamp_method_values[bs_idx, lag_idx, meth_idx, k_idx] = method
	                vamp_ks_values[bs_idx, lag_idx, meth_idx, k_idx] = k
	                vamp_iter_values[bs_idx, lag_idx, meth_idx, k_idx] = bs_idx
	
	
	# In[133]:
	
	
	ts = pd.DataFrame(data={'value': timescales_values.flatten(), 
	                       'lag': lag_values.flatten(),
	                       'num_its': its_values.flatten(), 
	                       'iteration': iter_values.flatten()})
	
	vamps = pd.DataFrame(data={'value': vamp_values.flatten(), 
	                          'lag': vamp_lag_values.flatten(), 
	                          'method': vamp_method_values.flatten(), 
	                          'k': vamp_ks_values.flatten(), 
	                          'iteration': vamp_iter_values.flatten()})
	
	hp = pd.DataFrame(data=bs_results[0]['hp'], index=[0])
	
	
	# In[135]:
	
	
	out_file = data_dir.joinpath(protein, hp_dir, 'summary.h5')
	
	ts.to_hdf(out_file, key='timescales')
	vamps.to_hdf(out_file, key='vamps')
	hp.to_hdf(out_file, key='hp')


# In[ ]:




