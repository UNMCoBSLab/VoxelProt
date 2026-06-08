
import numpy as np
from sklearn.neighbors import KDTree
import torch
TRP_AROMATIC = {"CD1", "CD2", "CG", "CE2", "CZ2", "CH2", "CZ3", "CE3"}
PHE_AROMATIC = {"CZ", "CE1", "CD1", "CG", "CD2", "CE2"}
TYR_AROMATIC = {"CZ", "CE2", "CD2", "CD1", "CE1", "CG"}

METALS = {"NA", "MG", "ZN", "MN", "CA", "FE"}
def _get_atom_name(atom):
    """
    Extract atom name from Biopython atom object or string-like atom record.

    Your old code used:
        str(atom)[6:][0:3]
    This keeps similar behavior, but also supports Biopython Atom objects.
    """
    if hasattr(atom, "get_name"):
        return atom.get_name().strip().upper()

    atom_str = str(atom)
    return atom_str[6:].strip().upper()
    
def scale_atom(protein_atoms, protein_atoms_res):
    '''
         assign each atom with a value:
         H:0, C_aliphatic:1, C_aromatic:2, N:3, O:4, P:5, S:6, Se:7, halogen(CL,IOD):8 and metal(Na,K,MG or ZN or MN or CA or FE):9
         # in TRP: CD1,CD2,CG,CE2,CZ2,CZ2,CH2,CZ3,CE3
         # in PHE: CZ,CE1,CD1,CG,CD2,CE2
         # in TYR: CZ,CE2,CD2,CD1,CE1,CG
         Args:
             protein_atoms(list): (N,3) atom type.
             protein_atoms_res (list): (N,1) the residue name of each atom.
          
         Returns:
             Tensor: (N,1) :atom type
    ''' 
    rv = [0 for _ in protein_atoms]

    for i, atom in enumerate(protein_atoms):
        atom_name = _get_atom_name(atom)
        res_name = str(protein_atoms_res[i]).strip().upper()

        s3 = atom_name[:3]
        s2 = atom_name[:2]
        s1 = atom_name[:1]

        # Aromatic carbon in aromatic residues
        if res_name == "TRP" and (s3 in TRP_AROMATIC or s2 in TRP_AROMATIC):
            rv[i] = 2
        elif res_name == "PHE" and (s3 in PHE_AROMATIC or s2 in PHE_AROMATIC):
            rv[i] = 2
        elif res_name == "TYR" and (s3 in TYR_AROMATIC or s2 in TYR_AROMATIC):
            rv[i] = 2

        # Metals
        elif s1 == "K" or s2 in METALS:
            rv[i] = 9

        # Halogens
        elif s2 == "CL" or s3 == "IOD":
            rv[i] = 8

        # Common elements
        elif s1 == "H":
            rv[i] = 0
        elif s1 == "C":
            rv[i] = 1
        elif s1 == "N":
            rv[i] = 3
        elif s1 == "O":
            rv[i] = 4
        elif s1 == "P":
            rv[i] = 5
        elif s1 == "S":
            rv[i] = 6
        elif s2 == "SE":
            rv[i] = 7

        # Default: keep as H-like/zero, same as your original behavior
        else:
            rv[i] = 0

    return rv

def getAtomType(protein_atoms, protein_atoms_res,protein_atoms_coor,surfacePoints):
    '''get the atom type for each atom
        Args:
            protein_atoms(list): (N,3) atom type.
            protein_atoms_res (list): (N,1) the residue name of each atom.
            surfacePoints(Tensor): (M,3) surface point coors.
        
        Returns:
            Tensor: (M,1) :the atomtype for each surface point
    '''
    # 1. Atom type for each protein atom
    atom_type = np.asarray(scale_atom(protein_atoms, protein_atoms_res),dtype=np.int64,)

    # 2. Move coordinates to CPU numpy for sklearn KDTree
    if isinstance(protein_atoms_coor, torch.Tensor):
        protein_atoms_coor_np = protein_atoms_coor.detach().cpu().numpy()
    else:
        protein_atoms_coor_np = np.asarray(protein_atoms_coor)

    if isinstance(surfacePoints, torch.Tensor):
        surface_np = surfacePoints.detach().cpu().numpy()
        device = surfacePoints.device
    else:
        surface_np = np.asarray(surfacePoints)
        device = "cpu"

    # 3. Nearest atom for each surface point
    kdt = KDTree(protein_atoms_coor_np, metric="euclidean")
    nearest = kdt.query(surface_np, k=1, return_distance=False)[:, 0]

    # 4. Vectorized one-hot encoding
    nearest_atom_types = torch.as_tensor(atom_type[nearest],dtype=torch.long,device=device)

    atom_features = torch.zeros(surface_np.shape[0],10,dtype=torch.float32,device=device)

    atom_features.scatter_(1, nearest_atom_types.view(-1, 1), 1.0)

    return atom_features
