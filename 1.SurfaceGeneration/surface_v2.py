#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from curvatures import *
from helpers import *
import torch
import pykeops
from pykeops.torch import LazyTensor
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from math import pi, sqrt
import numpy
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
inttensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
numpy = lambda x: x.detach().cpu().numpy()

def cal_atomtype_radius(protein_atoms):
    '''
          assign each atom with the Van der Waals radius(pm)
          [C,H,O,N,S,Se, P,CL,IOD,Na,K,MG,ZN,CA,FE]
          [0,1,2,3,4,5,  6,7,8, 9,10,11,12,13,14]
          [170,120,152,155,180,190,180,175,198,227,275,173,139,231,244]
          Args:
              protein_atoms(list): (N,3) atom type.

          Returns:
              Tensor: (N,1) :atom type
    ''' 
      
    atom_type=[0 for each in protein_atoms]
    for i in range(len(atom_type)):
        s3=str(protein_atoms[i])[6:][0:3]
        s2=str(protein_atoms[i])[6:][0:2]
        s1=str(protein_atoms[i])[6:][0:1]
  
        if s3=="IOD":
          atom_type[i]=8     
        elif s2=="NA":
          atom_type[i]=9  
        elif s2=="MG":
          atom_type[i]=11 
        elif s2=="ZN":
          atom_type[i]=12  
        elif s2=="CA":
          atom_type[i]=13  
        elif s2=="FE":
          atom_type[i]=14   
        elif s2=="CL":
          atom_type[i]=7 
        elif s2=="SE":
          atom_type[i]=5
        elif s1=="K":
          atom_type[i]=10    
        elif s1=="H":
          atom_type[i]=1
        elif s1=="C":
          atom_type[i]=0
        elif s1=="N":
          atom_type[i]=3
        elif s1=="O":
          atom_type[i]=2
        elif s1=="S":
          atom_type[i]=4
        elif s1=="P":
          atom_type[i]=6
        else:
          atom_type[i]=15

    atomtypes_protein=torch.zeros(len(protein_atoms),16)
    for ind in range(len(atomtypes_protein)):
        atomtypes_protein[ind][atom_type[ind]]=1
    return atomtypes_protein 

def sample(atoms, distance=1.05, sup_sampling=40):
    """generate a point cloud {X1,…Xnb}, where n is the number of atoms and b =20
        draw n*b points at random in the neighborhood of atoms
        Args:
          atoms (Tensor): (n,3) coordinates of the atoms.
          distance (float, optional): value of the level set to sample from
            the smooth distance function. Defaults to 1.05.
          sup_sampling=20  
    
        return:the coordinates of all sample points
      """
    num,dim=atoms.shape
    # sample n*b points at random in the neighborhood of our atoms
    sample_points = atoms[:, None, :] + 10 * distance * torch.randn(num, sup_sampling, dim).type_as(atoms)
    sample_points = sample_points.view(-1, dim)  # (N*B, D)
    return sample_points


def soft_distances(x,y,batch_x, batch_y,atomtypes,smoothness=0.01):
    """Computes a soft distance function to the atom centers of a protein.
      
    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        smoothness (float, optional): atom radii.Defaults to .01.
    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    # (N, 1, 3) atoms
    x_i = LazyTensor(x[:, None, :])
    # (1, M, 3) sampling points
    y_j = LazyTensor(y[None, :, :])
    # (N, M, 1) squared distances
    D_ij = ((x_i - y_j) ** 2).sum(-1)  

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    #D_ij.ranges = diagonal_ranges(batch_x, batch_y)
    if atomtypes is None:
        soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(-1)
    else:
        radius=torch.cuda.FloatTensor([170,120,152,155,180,190,180,175,198,227,275,173,139,231,244,100],device=x.device)
        #radius=torch.cuda.FloatTensor([170,120,152,155,180,190,180,100])
        radius=radius/radius.min()

        atomtype_radius=atomtypes * radius[None:]
        smoothness=torch.sum(smoothness * atomtype_radius,dim=1,keepdim=False) 
        smoothness_i=LazyTensor(smoothness[:,None,None])
        mean_smoothness=(-D_ij.sqrt()).exp().sum(0)
        mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
        mean_smoothness = (smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j)
        mean_smoothness = mean_smoothness.sum(0).view(-1)
        soft_dists = -mean_smoothness * ((-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)).view(-1)    
    return soft_dists


def normalVector(points, atoms, batch_x, batch_y, smoothness=0.01,atomtypes=None):
    """Computes a normal vector for each surface point, which points out always.
    
    Args:
        points: (N,3) surfacePoints.
        atoms: (M,3) protein atoms.
    Returns:
        Tensor: (N,3) normal vector for each surface point 
    """   
    p = points.detach()
    p.requires_grad = True
    dists = soft_distances(atoms,p,batch_x,batch_y,smoothness=smoothness,atomtypes=atomtypes)
    Loss = (1.0 * dists).sum()
    g = torch.autograd.grad(Loss, p)[0]
    return F.normalize(g, p=2, dim=-1) 


def createSurface(atoms, batch, variance=0.1, n_iter=4, distance=1.05, sup_sampling=20, smoothness=0.01,threshold=-0.7,reg=0.01,scales=[1.0],atomtypes=None):
    """generate the surface point model and assign each point with 
    a normal vector.
    The normal vector points toward the convex direction
    
    Args:    
    atoms (Tensor): (n,3) coordinates of the atoms.
    n_iter:Iterative loop,gradient descent along the potential 
      with respect to the positions samples. Defaults to 4. 
    batch (integer): like 100 batch. cited from:https://github.com/getkeops/keops/issues/73
    distance (float, optional): value of the level set to sample from
      the smooth distance function. Defaults to 1.05.   
    threshold(float): the shape index threshold. defaults to -0.7
    reg(float):default to 0.01
    scales:default to [1.0]. when smooth the local normals, scales will be used.
    """
    
    atoms=atoms.to(device="cuda")
    if atomtypes is None:
        pass
    else:
        atomtypes=atomtypes.to(device="cuda")
    # Batch vectors:
    num,dim=atoms.shape
    batch_atoms =torch.arange((num//batch)+1).view((num//batch)+1, 1).repeat(1, batch).view(batch*((num//batch)+1))[:num]
    batch_z = batch_atoms[:, None].repeat(1, sup_sampling).view(num * sup_sampling)

    # sampling points around each atom
    samples = sample(atoms, distance, sup_sampling)
    samples = samples.detach().contiguous()
    atoms = atoms.detach().contiguous()

    ##generating the surface with outliers  
    with torch.enable_grad():
        if samples.is_leaf:
            samples.requires_grad = True
            # Iterative loop: gradient descent along the potential with respect to the positions samples
        for it in range(n_iter):
            dists = soft_distances(atoms, samples, batch_x=batch_atoms, batch_y = batch_z, smoothness=smoothness,atomtypes=atomtypes)
            Loss = ((dists - distance) ** 2).sum()
            g = torch.autograd.grad(Loss, samples)[0]
            samples.data -= 0.5 * g  
      
    ##Removing the outliers as well as the points inside the proteins
    #1)Only keep the points which are reasonably close to the level set:
    dists = soft_distances(atoms,samples,batch_atoms,batch_z,atomtypes=atomtypes,smoothness=smoothness)
    margin = (dists-distance).abs()
    mask = margin < variance * distance
    #2) Remove insides points.
    zz = samples.detach()
    zz.requires_grad = True
    for it in range(n_iter):
        dists = soft_distances(atoms,zz,batch_x=batch_atoms, batch_y = batch_z, smoothness=smoothness,atomtypes=atomtypes)
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, zz)[0]
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
        zz = zz + 1.0 * distance * normals

    dists = soft_distances(atoms,zz,batch_atoms,batch_z,smoothness=smoothness,atomtypes=atomtypes)
    mask = mask & (dists > 1.5 * distance)
    remainPoints = samples[mask].contiguous().detach()
    batch_z= batch_z[mask].contiguous().detach()

      #find the normal vector for each surface point
    nor_vector=normalVector(remainPoints,atoms,batch_x=batch_atoms,batch_y=batch_z,smoothness=smoothness,atomtypes=atomtypes)
      
    #calculate the shape_index as well as the candidate_index
    if batch is None:
        batch_x=None
    else:
        batch_x=torch.arange((remainPoints.shape[0]//batch)+1).view((remainPoints.shape[0]//batch)+1, 1).repeat(1, batch).view(batch*((remainPoints.shape[0]//batch)+1))[:remainPoints.shape[0]] 
    shape_index=curvatures(remainPoints, scales=scales, batch=None, normals=nor_vector, reg=reg)

    candidate_index=(shape_index<threshold).nonzero().squeeze()

    # calcualte the oreented_nor_vector

    oriented_nor_vector=nor_vector.clone()
  
    for num in range(nor_vector.shape[0]):
        oriented_nor_vector[num]=oriented_nor_vector[num]*shape_index[num]
    #remainPoints=remainPoints-0.5 * nor_vector
    return shape_index, candidate_index, remainPoints, oriented_nor_vector

