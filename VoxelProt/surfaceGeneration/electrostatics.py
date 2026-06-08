"""
  Poisson Electrostatics.
  These values were normalized to be between -1 and 1.
"""
import numpy as np
import torch

# scale the electrostatics index
def map_to_range(arr, min_v=-1.0, max_v=1.0):
    arr = np.asarray(arr, dtype=np.float32)
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.float32)

    return np.interp(arr, (arr_min, arr_max), (min_v, max_v)).astype(np.float32)

def getPoiBol(g, surfacePoints):
    """
    Get Poisson-Boltzmann electrostatic value for each surface point.

    Args:
        g: gridData.core.Grid object from APBS
           g.grid:   [Nx, Ny, Nz] electrostatic grid
           g.origin: [3]
           g.delta:  [3]
        surfacePoints: torch.Tensor [M, 3] or numpy.ndarray [M, 3]

    Returns:
        torch.Tensor [M, 1], normalized electrostatic values in [-1, 1]
    """

    if isinstance(surfacePoints, torch.Tensor):
        device = surfacePoints.device
        surface_np = surfacePoints.detach().cpu().numpy()
    else:
        device = "cpu"
        surface_np = np.asarray(surfacePoints)

    # APBS grid
    grid = np.asarray(g.grid, dtype=np.float32)

    # Clip extreme electrostatic values, same logic as original build_dictionary()
    grid = np.clip(grid, -30.0, 30.0)

    origin = np.asarray(g.origin, dtype=np.float32)
    delta = np.asarray(g.delta, dtype=np.float32)

    # Convert real coordinates to grid indices
    # Original code used floor division: (x - x0) // delta_x
    indices = np.floor((surface_np - origin) / delta).astype(np.int64)

    ix,iy,iz = indices[:, 0],indices[:, 1],indices[:, 2]
    nx, ny, nz = grid.shape

    # Valid points inside the grid
    valid = ((ix >= 0) & (ix < nx) &(iy >= 0) & (iy < ny) &(iz >= 0) & (iz < nz))

    values = np.zeros(surface_np.shape[0], dtype=np.float32)

    # Direct vectorized grid lookup
    values[valid] = grid[ix[valid], iy[valid], iz[valid]]

    # Normalize to [-1, 1], same idea as original code
    values = map_to_range(values, -1.0, 1.0)

    values = torch.from_numpy(values).float().view(-1, 1).to(device)

    return values
