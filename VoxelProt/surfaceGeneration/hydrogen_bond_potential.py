"""
Hydrogen bond potential
These values range from −1 (optimal position for a hydrogen bond acceptor) 
to +1 (optimal position for a hydrogen bond donor).
"""
import numpy as np
import torch
from sklearn.neighbors import KDTree
from Bio.PDB import NeighborSearch
from Bio.PDB.vectors import Vector, calc_angle
from VoxelProt.surfaceGeneration.dictionary import *

def _to_numpy(x):
    """
    Convert torch tensor or numpy array to CPU numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
    
def computeAngleDeviation(a, b, c, theta):
  """Compute the absolute value of the deviation from theta
    Args:
        a, b and c: the corr of points
        theta: the ideal theta are based on data
        
    Returns:
        angle(pi)
  """
  return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)



def computeSatisfied_CO_HN(atoms):
    """Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
        Args:
            atoms(list):(N,3) the coor of each protein atoms
            
        Returns:
            satisfied_CO(set):
            satisfied_HN(set):
    """

    ns = NeighborSearch(atoms)
    satisfied_CO = set()
    satisfied_HN = set()

    for atom1 in atoms:
        res1 = atom1.get_parent()

        if atom1.get_id() != "O":
            continue

        # Need C atom for backbone carbonyl
        if "C" not in res1:
            continue

        neigh_atoms = ns.search(atom1.get_coord(), 3.0, level="A")

        for atom2 in neigh_atoms:
            if atom2.get_id() != "H":
                continue

            res2 = atom2.get_parent()

            # Different residues only
            if res2.get_id() == res1.get_id():
                continue

            # Need N atom for backbone N-H
            if "N" not in res2:
                continue

            angle_N_H_O_dev = computeAngleDeviation(
                res2["N"].get_coord(),
                atom2.get_coord(),
                atom1.get_coord(),
                16 * np.pi / 18,
            )

            angle_H_O_C_dev = computeAngleDeviation(
                atom2.get_coord(),
                atom1.get_coord(),
                res1["C"].get_coord(),
                15 * np.pi / 18,
            )

            if (
                angle_N_H_O_dev <= 3 * np.pi / 18
                and angle_H_O_C_dev <= 5.5 * np.pi / 18
            ):
                satisfied_CO.add(_residue_key(res1))
                satisfied_HN.add(_residue_key(res2))

    return satisfied_CO, satisfied_HN


def _residue_key(res):
    """
    Make a stable residue key using chain ID and residue ID.
    """
    chain_id = res.get_parent().get_id()
    if chain_id == "":
        chain_id = " "
    return chain_id, res.get_id()


def isPolarHydrogen(atom_name, res_type):
    """
    Determine whether a hydrogen atom is polar.
    """
    return res_type in polarHydrogens and atom_name in polarHydrogens[res_type]


def isAcceptorAtom(atom_name, res):
    """
    Determine whether an atom is an H-bond acceptor.
    """
    if atom_name.startswith("O"):
        return True

    if res.get_resname() == "HIS":
        if atom_name == "ND1" and "HD1" not in res:
            return True
        if atom_name == "NE2" and "HE2" not in res:
            return True

    return False


def computeAnglePenalty(angle_deviation, std_dev):
    """
    Convert angular deviation to penalty score.
    """
    return max(0.0, 1.0 - (angle_deviation / std_dev) ** 2)


def computeChargeHelper(atom_name, res, v):
    """
    Compute hydrogen-bond potential contribution for one surface point.

    Args:
        atom_name: nearest atom name
        res: Bio.PDB Residue object
        v: surface point coordinate, numpy array [3]

    Returns:
        float
    """

    res_type = res.get_resname()

    # Donor: polar hydrogen
    if isPolarHydrogen(atom_name, res_type):
        if atom_name not in donorAtom:
            return 0.0

        donor_atom_name = donorAtom[atom_name]

        if donor_atom_name not in res:
            return 0.0

        a = res[donor_atom_name].get_coord()  # donor heavy atom, N/O
        b = res[atom_name].get_coord()        # H atom

        angle_deviation = computeAngleDeviation(a, b, v, np.pi)
        angle_penalty = computeAnglePenalty(angle_deviation, np.pi)

        return 1.0 * angle_penalty

    # Acceptor
    if isAcceptorAtom(atom_name, res):
        if atom_name not in acceptorAngleAtom:
            return 0.0

        angle_atom_name = acceptorAngleAtom[atom_name]

        if angle_atom_name not in res:
            return 0.0

        b = res[atom_name].get_coord()
        a = res[angle_atom_name].get_coord()

        angle_deviation = computeAngleDeviation(a, b, v, 2 * np.pi / 3)
        angle_penalty = computeAnglePenalty(angle_deviation, np.pi)

        return -1.0 * angle_penalty

    return 0.0


def computeCharge(surfacePoints, protein_atoms_coor, protein_atoms):
    """
    Compute hydrogen-bond potential for all surface points.

    Args:
        surfacePoints: torch.Tensor [M, 3] or numpy.ndarray [M, 3]
        protein_atoms_coor: torch.Tensor [N, 3] or numpy.ndarray [N, 3]
        protein_atoms: list of Bio.PDB Atom objects

    Returns:
        torch.Tensor [M, 1]
    """

    # Preserve output device
    if isinstance(surfacePoints, torch.Tensor):
        out_device = surfacePoints.device
        surface_np = surfacePoints.detach().cpu().numpy()
    else:
        out_device = "cpu"
        surface_np = np.asarray(surfacePoints)

    protein_coor_np = _to_numpy(protein_atoms_coor[:, 0:3])

    # Backbone satisfied H-bond atoms
    satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(protein_atoms)

    # Build residue dictionary once
    residues = {}
    for atom in protein_atoms:
        res = atom.get_parent()
        residues[_residue_key(res)] = res

    # Nearest atom for each surface point
    kdt = KDTree(protein_coor_np, metric="euclidean")
    nearest = kdt.query(surface_np, k=1, return_distance=False)[:, 0]

    charge = np.zeros(surface_np.shape[0], dtype=np.float32)

    for i, atom_idx in enumerate(nearest):
        atom = protein_atoms[int(atom_idx)]
        atom_name = atom.get_id()
        res = atom.get_parent()
        res_key = _residue_key(res)

        # Ignore already satisfied backbone atoms
        if atom_name == "H" and res_key in satisfied_HN:
            continue

        if atom_name == "O" and res_key in satisfied_CO:
            continue

        charge[i] = computeChargeHelper(
            atom_name,
            residues[res_key],
            surface_np[i],
        )

    return torch.from_numpy(charge).view(-1, 1).to(out_device)
