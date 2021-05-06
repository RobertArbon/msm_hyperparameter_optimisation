from typing import Optional

from pyemma.coordinates.data._base.datasource import DataSource
import mdtraj as md
import numpy as np


def add_phipsi_dihdedrals(reader: DataSource) -> DataSource:
    reader.featurizer.add_backbone_torsions(cossin=True)
    return reader


def num_contacts(reader: DataSource) -> int:
    traj = md.load_frame(reader.filenames[0], top=reader.featurizer.topology, index=0)
    _, ix = md.compute_contacts(traj, contacts='all')
    return ix.shape[0]


def add_contacts(reader: DataSource, cutoff: float, scheme: Optional[str] = 'closest-heavy') -> DataSource:
    dim = num_contacts(reader)

    def _contacts(traj):
        feat, ix = md.compute_contacts(traj, contacts='all', scheme=scheme)
        feat = ((feat <= cutoff)*1).astype(np.float32)
        return feat

    reader.featurizer.add_custom_func(_contacts, dim=dim)
    return reader
