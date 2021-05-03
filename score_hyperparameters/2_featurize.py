"""
for a system (e.g., protein) saves features.
"""
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import pyemma
import mdtraj as md

import constants as cons


def get_input_trajs_top(glob_str: str) -> Dict[str, List[Path]]:
    trajs = list(Path('/').glob(f"{glob_str}/*.xtc"))
    top = list(Path('/').glob(f"{glob_str}/*.pdb"))[0]
    trajs.sort()
    assert trajs, 'no trajectories found'
    assert top, 'no topology found'
    return {'top': top, 'trajs': trajs}


def get_hyperparameters(path: str) -> pd.DataFrame:
    hps = pd.read_hdf(path)
    return hps


def create_reader(traj_top_paths):
    trajs = [str(x) for x in traj_top_paths['trajs']]
    top = str(traj_top_paths['top'])
    reader = pyemma.coordinates.source(trajs[:10], top=top)
    return reader


def create_name(hp: Dict) -> str:
    feature_keys = [x for x in hp.keys() if x.startswith('feature')]
    fname_list = []
    for key in feature_keys:
        elements = key.split('__')
        if 'value' == elements[1] or hp['feature__value'] == elements[1]:
            fname_list.append(f"{hp[key]}")
    name = f"{'_'.join(fname_list)}.h5"
    return name


def add_phipsi_dihdedrals(reader: pyemma.coordinates.source) -> pyemma.coordinates.source:
    reader.featurizer.add_backbone_torsions(cossin=True)
    return reader


def num_contacts(reader: pyemma.coordinates.source) -> int:
    traj = md.load_frame(reader.filenames[0], top=reader.featurizer.topology, index=0)
    _, ix = md.compute_contacts(traj, contacts='all')
    return ix.shape[0]


def add_contacts(reader: pyemma.coordinates.source, cutoff: float, scheme: Optional[str] = 'closest-heavy') -> pyemma.coordinates.source:
    dim = num_contacts(reader)

    def _contacts(traj):
        feat, ix = md.compute_contacts(traj, contacts='all', scheme=scheme)
        feat = ((feat <= cutoff)*1).astype(np.float32)
        return feat

    reader.featurizer.add_custom_func(_contacts, dim=dim)
    return reader


def create_features(hp_dict: Dict, traj_top_paths: Dict[str, Path], output_path: Path) -> None:
    feature_name = create_name(hp_dict)
    feature_path = output_path.joinpath(feature_name)
    reader = create_reader(traj_top_paths)

    feature = hp_dict['feature__value']
    if feature == 'phipsi_dihedrals':
        reader = add_phipsi_dihdedrals(reader)
    elif feature == 'contacts':
        cutoff = float(hp_dict['feature__contacts__center']*cons.UNITS['feature__contacts__center'])
        reader = add_contacts(reader, cutoff=cutoff)

    reader.write_to_hdf5(str(feature_path), group=feature_name,  overwrite=True,
                      stride=cons.STRIDE, chunksize=1000, h5_opt=dict(compression=32001, chunks=True, shuffle=True))


def filter_unique_hps(df: pd.DataFrame) -> pd.DataFrame:
    unique_ixs = []
    unique_names = []
    for i, row in df.iterrows():
        name = create_name(row.to_dict(into=dict))
        if not name in unique_names:
            unique_names.append(name)
            unique_ixs.append(i)
    return df.iloc[unique_ixs, :]


if __name__ == '__main__':
    hps = get_hyperparameters(Path(cons.NAME).joinpath('hp_sample.h5'))
    traj_top_paths = get_input_trajs_top(cons.INPUT_TRAJ_GLOB)
    unique_hps = filter_unique_hps(hps)
    for i, row in unique_hps.iterrows():
        create_features(row.to_dict(into=dict), traj_top_paths, Path(cons.NAME))

