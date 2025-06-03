"""
  Poisson Electrostatics.
  These values were normalized to be between -1 and 1.
"""
import numpy as np
import torch
import numpy as np
import torch

def build_dictionary(g):
  """get all coordinates of grid
    Args:
        g(gridData.core.Grid): the gridData from APBS.       
    Returns:
        dictionary: k is the coordinates and value is the corresponding value
  """
  #elec_corrs store all coordinates of grid
  elec_corrs={}
  for x in range(np.shape(g.grid)[0]):
    for y in range(np.shape(g.grid)[1]):
      for z in range(np.shape(g.grid)[2]):
        if g.grid[x,y,z]>=30:
          elec_corrs[(x,y,z)]=float(30)
        elif g.grid[x,y,z]<=-30:
          elec_corrs[(x,y,z)]=float(-30)
        else:
          elec_corrs[(x,y,z)]=float(g.grid[x,y,z])
  return elec_corrs


# scale the electrostatics index
def map_to_range(arr,min_v,max_v):
  return np.interp(arr,(arr.min(), arr.max()), (min_v, max_v))

def getPoiBol(g,surfacePoints):
  """get the Poisson Electrostatic value within [-1,1] for each surface point
    Args:
        g(gridData.core.Grid): the gridData from APBS. 
        surfacePoints(Tensor): (M,3) surface point coors.
    Returns:
        Tensor(M,1): store all coordinates of grid+its corresponding eletrostatic value
  """
  elec_corrs=build_dictionary(g)
  PoiBol_features=torch.zeros(surfacePoints.size()[0],1)
  x0,y0,z0=g.origin[0],g.origin[1],g.origin[2]
  delta_x,delta_y,delta_z=g.delta[0],g.delta[1],g.delta[2]
  for i in range(surfacePoints.shape[0]):
    x, y, z = surfacePoints[i][0].item(),surfacePoints[i][1].item(),surfacePoints[i][2].item()
    try:
      PoiBol_features[i]=elec_corrs[((x-x0)//delta_x,(y-y0)//delta_y,(z-z0)//delta_z)]
    except:
      PoiBol_features[i]=0.0
  PoiBol_features=map_to_range(PoiBol_features,-1,1)
  return torch.tensor(PoiBol_features)
