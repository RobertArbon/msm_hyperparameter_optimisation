"""
for a system (e.g., protein) saves features.
"""
from typing import Dict, List, Mapping
from pathlib import Path

import pandas as pd

import setup as cons
from featurizers import *


def get_input_trajs_top() -> Dict[str, List[Path]]:
    glob_str = cons.INPUT_TRAJ_GLOB
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
    reader = pyemma.coordinates.source(trajs, top=top)
    return reader


def create_name(hp: Mapping) -> str:
    feature_keys = [x for x in hp.keys() if x.startswith('feature')]
    fname_list = []
    for key in feature_keys:
        elements = key.split('__')
        if 'value' == elements[1] or hp['feature__value'] == elements[1]:
            fname_list.append(f"{hp[key]}")
    name = f"{'_'.join(fname_list)}.h5"
    return name


def create_features(hp_dict: Mapping, traj_top_paths: Dict[str, List[Path]], output_path: Path) -> None:
    feature_name = create_name(hp_dict)
    feature_path = output_path.joinpath(feature_name)
    reader = create_reader(traj_top_paths)

    feature = hp_dict['feature__value']
    if feature == 'phipsi_dihedrals':
        reader = add_phipsi_dihdedrals(reader)
    elif feature == 'contacts':
        cutoff = float(hp_dict['feature__contacts__center']*cons.UNITS['feature__contacts__center'])
        reader = add_contacts(reader, cutoff=cutoff)

    reader.write_to_hdf5(str(feature_path), group=feature,  overwrite=True,
                      stride=cons.STRIDE, chunksize=1000)


def filter_unique_hps(df: pd.DataFrame) -> pd.DataFrame:
    unique_ixs = []
    unique_names = []
    for i, row in df.iterrows():
        name = create_name(row.to_dict(into=dict))
        if not name in unique_names:
            unique_names.append(name)
            unique_ixs.append(i)
    return df.iloc[unique_ixs, :]


def create_ouput_directory() -> Path:
    path = Path(cons.NAME)
    path.mkdir(exist_ok=True)
    return path


def estimate_count_matrices(hp: Mapping, traj_top_paths: Dict[str, List[Path]], output_dir: Path) -> None:
    pass




def main(hp_path: str) -> None:
    hps = get_hyperparameters(hp_path)
    output_dir = create_ouput_directory()
    traj_top_paths = get_input_trajs_top()
    unique_hps = filter_unique_hps(hps)

    for i, row in unique_hps.iterrows():
        estimate_count_matrices(row.to_dict(), traj_top_paths, output_dir)
        # create_features(row.to_dict(into=dict), traj_top_paths, output_dir)


if __name__ == '__main__':
    main('./hp_sample.h5')
