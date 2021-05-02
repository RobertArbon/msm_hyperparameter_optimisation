HP_SPACE = {'features__values': ['phipsi_dihedrals', 'contacts'],
            'tica__dims': [1, 10], 'tica__lag': [1, 100],
            'cluster__n_clusters': [10, 1000], 'cluster__max_iter': [1000, 1000],
            'contacts__center': [3.0, 15.0]}

OUTPUT_TRAJ_DIR = './Features'
INPUT_TRAJ_DIR = '/Volumes/REA/MD/12FF'
NUM_HPS = 200
NAME = 'experiment'
