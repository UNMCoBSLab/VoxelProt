import numpy as np
from sklearn.neighbors import KDTree
import torch
from VoxelProt.surfaceGeneration.dictionary import *
def _to_numpy(x):
    """
    Convert torch tensor or array-like object to CPU numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def scale_kd(protein_atoms_res, device="cpu", dtype=torch.float32):
    """scale the Kyte-Doolittle index from[-4.5,4.5] to [-1,1]
    and combine the corr of each atom with their scaled Kyte-Doolittle index
    Args:
        protein_atoms_res (list): (N,1) the residue name of each atom.
        
    Returns:
        Tensor: (N,1) :Kyte-Doolittle index
    """

    kd_values = np.array([float(Kyte_Doolittle[str(res).strip().upper()]) for res in protein_atoms_res],dtype=np.float32)
    kd_scaled = (2.0 * (kd_values + 4.5) / 9.0) - 1.0

    return torch.as_tensor(kd_scaled, dtype=dtype, device=device).view(-1, 1)




def getKD(protein_atoms_coor,protein_atoms_res,surfacePoints):
    """get the Kyte-Doolittle index within [-1,1] for each surface point
    Args:
        protein_atoms_coor (Tensor): (N,3) atom coors.
        protein_atoms_res (list): (N,1) the residue name of each atom.
        surfacePoints(Tensor): (M,3) surface point coors.
        
    Returns:
        Tensor: (M,1) :the Kyte-Doolittle index within [-1,1] for each surface point
    """
    if isinstance(surfacePoints, torch.Tensor):
        out_device = surfacePoints.device
        out_dtype = surfacePoints.dtype if surfacePoints.dtype.is_floating_point else torch.float32
    else:
        out_device = "cpu"
        out_dtype = torch.float32

    protein_np = _to_numpy(protein_atoms_coor[:, 0:3]).astype(np.float32)
    surface_np = _to_numpy(surfacePoints[:, 0:3]).astype(np.float32)

    kd_values = np.array(
        [float(Kyte_Doolittle[str(res).strip().upper()]) for res in protein_atoms_res],
        dtype=np.float32,
    )
    kd_scaled = (2.0 * (kd_values + 4.5) / 9.0) - 1.0

    # Nearest atom for each surface point
    kdt = KDTree(protein_np, metric="euclidean")
    nearest = kdt.query(surface_np, k=1, return_distance=False)[:, 0]
    kd_features = kd_scaled[nearest].astype(np.float32)
    return torch.as_tensor(kd_features, dtype=out_dtype, device=out_device).view(-1, 1)

