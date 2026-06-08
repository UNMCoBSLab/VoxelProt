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
    scales = torch.as_tensor(scales,dtype=vertices.dtype,device=vertices.device)

    centers = vertices

    # KeOps symbolic tensors
    x_i = LazyTensor(vertices[:, None, :])     # [N, 1, 3]
    y_j = LazyTensor(centers[None, :, :])      # [1, N, 3]
    n_j = LazyTensor(normals[None, :, :])      # [1, N, 3]
    s = LazyTensor(scales[None, None, :])      # [1, 1, S]

    # Squared Euclidean distances
    D_ij = ((x_i - y_j) ** 2).sum(-1)          # [N, N, 1]

    # Gaussian kernel for each scale
    K_ij = (-D_ij / (2 * s ** 2)).exp()        # [N, N, S]

    if batch is not None:
        K_ij.ranges = diagonal_ranges(batch, batch)

    # Weighted sum of neighboring normals
    smoothed = K_ij.tensorprod(n_j).sum(dim=1) # [N, S*3]
    smoothed = smoothed.view(-1, len(scales), 3)

    smoothed = F.normalize(smoothed, p=2, dim=-1)

    return smoothed


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

    x_i = LazyTensor(vertices.view(N, 1, 3))
    x_j = LazyTensor(vertices.view(1, N, 3))
    x_diff = x_j - x_i

    curvature = []

    for scale_index, scale in enumerate(scales):
        normals_scale = smoothed_normals[:, scale_index, :].contiguous()
        uv = tangents[:, scale_index, :, :].contiguous()

        n_i = LazyTensor(normals_scale.view(N, 1, 3))
        n_j = LazyTensor(normals_scale.view(1, N, 3))
        uv_i = LazyTensor(uv.view(N, 1, 6))

        # Pseudo-geodesic squared distance
        d2_ij = (x_diff ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)

        # Gaussian window
        window_ij = (-d2_ij / (2 * (scale ** 2))).exp()

        # Project coordinate and normal differences onto tangent basis
        P_ij = uv_i.matvecmult(x_diff)
        Q_ij = uv_i.matvecmult(n_j - n_i)

        # Concatenate P and Q
        PQ_ij = P_ij.concat(Q_ij)

        # Local covariance terms
        PPt_PQt_ij = P_ij.tensorprod(PQ_ij)
        PPt_PQt_ij = window_ij * PPt_PQt_ij

        # Batch-aware reduction
        PPt_PQt_ij.ranges = ranges
        PPt_PQt = PPt_PQt_ij.sum(1)

        # Reshape into two 2x2 matrices
        PPt_PQt = PPt_PQt.view(N, 2, 2, 2)
        PPt = PPt_PQt[:, :, 0, :]
        PQt = PPt_PQt[:, :, 1, :]

        # Ridge regularization
        PPt[:, 0, 0] += reg
        PPt[:, 1, 1] += reg

        # Solve local linear system
        S = torch.linalg.solve(PPt, PQt)

        a = S[:, 0, 0]
        b = S[:, 0, 1]
        c = S[:, 1, 0]
        d = S[:, 1, 1]

        mean_curvature = a + d
        gauss_curvature = a * d - b * c

        curvature.append(mean_curvature.clamp(-1, 1))
        curvature.append(gauss_curvature.clamp(-1, 1))

    curvature = torch.stack(curvature, dim=-1)

    mean_curvature = curvature[:, 0]
    gauss_curvature = curvature[:, 1]

    elem = torch.square(mean_curvature) - gauss_curvature
    elem[elem <= 0] = 1e-8

    k1 = mean_curvature + torch.sqrt(elem)
    k2 = mean_curvature - torch.sqrt(elem)

    shape_index = torch.arctan((k1 + k2) / (k1 - k2)) * (2 / torch.pi)

    return shape_index
