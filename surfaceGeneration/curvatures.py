"""This module is used to calculate the shape index
"""
from VoxelProt.surfaceGeneration.helpers import *
import torch
import pykeops
from pykeops.torch import LazyTensor
import torch.nn.functional as F
import numpy

def smooth_normals(vertices, normals, scales=[1.0], batch=None):
    """This function is used to smooth the local normals,
      which is the average of all normals in a ball of size scales   
      Args:
      vertices(tensor):(N,3) the coordinates of all surface points  
      normals (tensor):(N,3) the normals of all surface points 
      scales([float]): the scale of the ball
      batch(tensor):batch operator 
    """
    #change scales into tensor type
    scales = torch.Tensor(scales).type_as(vertices)

    #the centers of balls are all vertices
    centers = vertices

    # Normal of a vertex:
    x = LazyTensor(vertices[:, None, :])  
    y = LazyTensor(centers[None, :, :])  
    z = LazyTensor(normals[None, :, :])  
    s = LazyTensor(scales[None, None, :])  

    D_ij = ((x - y) ** 2).sum(-1)  
    K_ij = (-D_ij / (2 * s ** 2)).exp()  

    if batch is not None:
        K_ij.ranges = diagonal_ranges(batch,batch)

    normals = (K_ij.tensorprod(z)).sum(dim=1)  
    normals = normals.view(-1, len(scales), 3)
    normals = F.normalize(normals, p=2, dim=-1)  
    
    return normals

def tangent_vectors(normals):
    """This function is used to calculate the tangent plane
      Args:
      normals(tensor):(N*3) the normal vectors of each vertice
      output:each point has two tangent vector, orthogonal to each other
    """
    x, y, z = normals[...,0], normals[...,1], normals[...,2]
    # if z>=0,signz=1; if z<0,signz=-1; 
    signz = (2 * (z >= 0)) - 1.0  
    a = -1 / (signz + z)
    b = x * y * a
    #get two tangent vectors
    tangent_vectors = torch.stack((1+signz*a*x*x,signz*b,-signz*x,b,signz+a*y*y,-y),dim=-1)
    #change the shape
    tangent_vectors = tangent_vectors.view(tangent_vectors.shape[:-1] + (2, 3)) 
    return tangent_vectors

def curvatures(vertices, normals, scales=[1.0], batch=None, reg=0.01):
    """This function is used to calculate the shape index
    Args:
      vertices(tensor):(N,3) the coordinates of all surface points  
      normals (tensor):(N,3) the normals of all surface points 
      scales([float]): the scale of the ball
      batch(tensor):batch operator 
      reg(float):a small ridge regression.default to 0.01
    """

    # Number of points, number of scales:
    N, S = vertices.shape[0], len(scales)
    ranges = diagonal_ranges(batch)

    # smooth the normals:
    smoothed_normals=smooth_normals(vertices, normals=normals, scales=scales, batch=batch) 

    # calculate the tangent plane of each point:
    tangents = tangent_vectors(smoothed_normals)  

    curvature = []

    for ind, scale in enumerate(scales):
        normals = smoothed_normals[:, ind, :].contiguous() 
        uv = tangents[:, ind, :, :].contiguous() 

        # Encode as symbolic tensors:
        # Points:
        x_i = LazyTensor(vertices.view(N, 1, 3))
        x_j = LazyTensor(vertices.view(1, N, 3))
        # Normals:
        n_i = LazyTensor(normals.view(N, 1, 3))
        n_j = LazyTensor(normals.view(1, N, 3))
        # Tangent bases:
        uv_i = LazyTensor(uv.view(N, 1, 6))

        # Pseudo-geodesic squared distance:
        d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  
        # Gaussian window:
        window_ij = (-d2_ij / (2 * (scale ** 2))).exp()  

        # Calculate the P and Q
        P_ij = uv_i.matvecmult(x_j - x_i)  
        Q_ij = uv_i.matvecmult(n_j - n_i) 
        # Concatenate P and Q
        PQ_ij = P_ij.concat(Q_ij)  

        # Covariances
        PPt_PQt_ij = P_ij.tensorprod(PQ_ij)  
        PPt_PQt_ij = window_ij * PPt_PQt_ij  

        # Reduction - with batch support:
        PPt_PQt_ij.ranges = ranges
        PPt_PQt = PPt_PQt_ij.sum(1)  

        # Reshape to get the two covariance matrices:
        PPt_PQt = PPt_PQt.view(N, 2, 2, 2)
        PPt, PQt = PPt_PQt[:, :, 0, :], PPt_PQt[:, :, 1, :] 

        # Add a small ridge regression:
        PPt[:, 0, 0] += reg
        PPt[:, 1, 1] += reg

        S = torch.linalg.solve(PPt, PQt)
        a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]  

        # mean curvature is the trace and gauss curvature is the determinant
        mean_curvature = a + d
        gauss_curvature = a * d - b * c
        curvature += [mean_curvature.clamp(-1, 1), gauss_curvature.clamp(-1, 1)]

    curvature = torch.stack(curvature, dim=-1)
    
    mean_curvature = curvature[:,0]
    gauss_curvature = curvature[:,1]
    elem = torch.square(mean_curvature) - gauss_curvature  
    elem[elem<=0] = 1e-8
    k1 = mean_curvature + torch.sqrt(elem)
    k2 = mean_curvature - torch.sqrt(elem)
  
    # Compute the shape index 
    shape_index = torch.arctan( (k1+k2)/(k1-k2) )*(2/torch.pi)
    return shape_index
