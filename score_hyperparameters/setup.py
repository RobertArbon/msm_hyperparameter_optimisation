import numpy as np
#
# EXPERIMENT PARAMETERS
#
NAME = '1FME'
INPUT_TRAJ_GLOB = 'Volumes/REA/MD/12FF/DESRES-Trajectory_1FME-*-protein/1FME-*-protein'
NUM_TRIALS = 5
BS_SAMPLES = 2

#
# HYPER-PARAMETERS
# keep values integers. If you need floats, change the units dict.
# MDTraj deals with nm no angstroms!
#
HP_SPACE = {'feature__value': ['phipsi_dihedrals', 'contacts'],
            'tica__dim': [1, 10], 'tica__lag': [1, 100],
            'cluster__k': [10, 1000], 'cluster__max_iter': [1000, 1000],
            'feature__contacts__center': [30, 150]}
UNITS = {k: 1 for k in HP_SPACE.keys()}
UNITS['feature__contacts__center'] = 0.01

#
# DATA PARAMETERS
#
STRIDE = 1

#
# MSM PARAMETERS
#
LAGS = np.arange(2, 51, 2)
MSM_LAG = 10