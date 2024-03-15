#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
from sklearn.neighbors import KDTree
import torch
from dictionary import *
test="gethydropath_info"
def scale(protein_atoms_coor,protein_atoms_res):
  """scale the Kyte-Doolittle index from[-4.5,4.5] to [-1,1]
    and combine the corr of each atom with their scaled Kyte-Doolittle index
    Args:
        protein_atoms_coor (Tensor): (N,3) atom coors.
        protein_atoms_res (list): (N,1) the residue name of each atom.
        
    Returns:
        Tensor: (N,1) :Kyte-Doolittle index
  """

  temp=numpy.array([float(Kyte_Doolittle[item]) for item in protein_atoms_res])
  result=torch.tensor((2*(temp+4.5)/9)-1)
  return result.view(result.size()[0],1)



def getKD(protein_atoms_coor,protein_atoms_res,surfacePoints):
  """get the Kyte-Doolittle index within [-1,1] for each surface point
    Args:
        protein_atoms_coor (Tensor): (N,3) atom coors.
        protein_atoms_res (list): (N,1) the residue name of each atom.
        surfacePoints(Tensor): (M,3) surface point coors.
        
    Returns:
        Tensor: (M,1) :the Kyte-Doolittle index within [-1,1] for each surface point
  """
  all_KD=scale(protein_atoms_coor,protein_atoms_res)
  KD_features=torch.zeros(surfacePoints.size()[0],1)

  kdt = KDTree(protein_atoms_coor.detach().numpy(), metric='euclidean')
  nearest=kdt.query(surfacePoints,1,return_distance=False)[:,0]
  for ind in range(len(surfacePoints)):
    KD_features[ind]=all_KD[nearest[ind]].item()
  
  """
  ## del starting: after showing
  import matplotlib.pyplot as plt
  import random
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure(figsize=(14, 12))
  ax = fig.add_subplot(projection='3d')
  x=surfacePoints.detach().cpu().numpy()[:,0]
  y=surfacePoints.detach().cpu().numpy()[:,1]
  z=surfacePoints.detach().cpu().numpy()[:,2]
  t=KD_features.detach().cpu().numpy()[:,0]
  p=ax.scatter(x,y,z,c=t)
  plt.set_cmap('bwr')
  fig.colorbar(p,ax=ax)
  plt.show()

  ## del ending 
  """

  return KD_features

