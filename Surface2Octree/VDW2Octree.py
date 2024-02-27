#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from shufflekeytable import *
class VDWOctree:
  """ Builds an octree from an input point cloud( in range [-1,1]).
  Args:
    depth (int): The octree depth. default to 4.
    full_depth (int): The octree layers which are shorter than full_depth are forced to be full.default to 2
    batch_size (int): The octree batch size. default to 1
    device (str): `cpu` or `gpu`. default: `gpu`
  """

  def __init__(self, device, depth=4, full_depth = 2, batch_size = 1):
    self.depth = depth
    self.full_depth = full_depth
    self.batch_size = batch_size
    self.device = device
    #Initialize the Octree status and construct several lookup tables
    self.initialize()

  def initialize(self):
    """ Initialize the Octree status and constructs several lookup tables.
    """

    # shuffle keys and labels of each octant in each octree layer
    self.shuffle_keys = [None] * (self.depth + 1)
    self.label = [None] * (self.depth + 1)

    # neighs is to store the suffle key of each octant
    # Specifically, suppose there is a octant "a", and there is a filter(neighborhood) 3*3
    # put a in the center of this 3*3 filter
    # record the shuffle key of each neighbors from down to up, from left to right
    # if one neighbor is empty, assign it to -1
    # in this way, the 14th cell is the filter core as well as is the octant a,
    # So, its value should be the shuffle key of a 
    self.neighs = [None] * (self.depth + 1)

    # to store the features, normals and coordinates of each octant
    self.energy = [None] * (self.depth + 1)
    self.points = [None] * (self.depth + 1)

    # octants numbers in each octree layers
    self.num = torch.zeros((self.depth + 1), dtype=torch.int32)
    # non-emply octants numbers in each octree layers
    self.nempty_num = torch.zeros((self.depth + 1), dtype=torch.int32)

    self.parent_table,self.child_table=self.search_neigh_table()

  def build_octree(self, point_cloud):
    """ Builds an octree from a point cloud.
    Args:
      point_cloud (Points): The input point cloud. in range [-1,1]
    """
    self.device = point_cloud.device

    # normalize points from [-1, 1] to [0, 2^depth]
    points = ((point_cloud.psedoAtom + 1.0) * (2 ** (self.depth - 1)))

    # layer 0 to full_layer: the octree is full in these layers
    for d in range(self.full_depth+1):
      n = 1 << (3 * d)
      self.shuffle_keys[d] = torch.arange(n, dtype=torch.long, device=self.device)
      self.label[d] = torch.arange(n, dtype=torch.int32, device=self.device)
      self.num[d] = n
      self.nempty_num[d] = n

    # for each surface point, get its shuffle key value in the finest level
    # and sort it
    sk = xyz2shufflekey(points[:, 0], points[:, 1], points[:, 2], depth=self.depth)
    keys, idx, counts = torch.unique(sk,return_inverse=True,return_counts=True)
 
    # layer depth_ to full_layer_
    for d in range(self.depth, self.full_depth, -1):
      # compute parent key
      parent_key = keys >> 3
      parent_key, parent_idx= torch.unique_consecutive(parent_key, return_inverse=True)

      # get the children key based on the parent key
      key = (parent_key.unsqueeze(-1) * 8).to(self.device) + torch.arange(8, device=self.device)
      self.shuffle_keys[d] = key.view(-1)
      self.num[d] = key.numel()
      self.nempty_num[d] = keys.numel()

      # update the label
      addr = (parent_idx << 3) | (keys % 8)
      l = -torch.ones(self.num[d].item(), dtype=torch.int32, device=self.device)
      l[addr] = torch.arange(self.nempty_num[d], dtype=torch.int32, device=self.device)
      self.label[d] = l

      # update the shuffle key for the next iteration
      keys = parent_key


    # create the label for the layer full_layer(d=2),
    L = -torch.ones_like(self.label[self.full_depth])
    L[keys] = torch.arange(keys.numel(), dtype=torch.int32, device=self.device)
    self.label[self.full_depth] = L
    self.nempty_num[self.full_depth] = keys.numel()

    # average the input signal for the last octree layer(d=4)
    # points, features are the average value of all points in each octant
    # normal vector are the normalized average value of all points in each octant
    points = self.scatter_addition(points, idx, dim=0) .to(self.device)
    self.points[self.depth] = points / counts.unsqueeze(1).to(self.device)

    energy = self.scatter_addition(point_cloud.vdws, idx, dim=0).to(self.device)
    self.energy[self.depth] = energy / counts.unsqueeze(1).to(self.device)

    return


  def get_input_signal(self):
    """ Get the input feature signals in the finest level.
       Output is a N*M (tensor): N is each occupied octant. M is the normals and chemical features
       [normal_x, normal_y, normal_z, hydropathy, electrostatics, hydrogenpotential]
    """
    # energy features
    features = []
    features.append(self.energy[self.depth])

    #chemical featuers
    #features.append(self.features[self.depth])
    return torch.cat(features, dim=1)

  def build_neigh(self):
    """ In the depth-th layer, let each octant as the core, then create a 3x3x3 neighbors for each core.
       If one neigh is missing ,set it to -1
       otherwise, octants are sorted by their shuffle key and get a index value. And the neighbor are represented by this index vaue
       output: implement the self.neighs
    """
    for depth in range(1, self.depth+1):
      if depth <= self.full_depth:
        n = 1 << (3 * depth)
        shufflekey = torch.arange(n, dtype=torch.long, device=self.device)
        x, y, z, _ = shufflekey2xyz(shufflekey, depth)
        xyz = torch.stack([x, y, z], dim=-1)  # (N,  3)
        xyz = xyz.unsqueeze(1) + self.meshgrid(min=-1, max=1)     # (N, 27, 3)
        #reshape
        xyz = xyz.view(-1, 3)      # (N*27, 3)
        neigh = xyz2shufflekey(xyz[:, 0], xyz[:, 1], xyz[:, 2], depth=depth)# (N*27,)
        # using batch to calculate the neigh
        batch_size = torch.arange(self.batch_size, dtype=torch.int32, device=self.device)
        neigh = neigh + batch_size.unsqueeze(1) * n  # (batch_size, N*27)=(N*27,) + (batch_size, 1) 
        # empty neighbors are assigned to -1
        empty_neigh = torch.logical_or((xyz < 0).any(1), (xyz >= (1 << depth)).any(1))    
        neigh[:, empty_neigh] = -1
        # reshape the neigh block shape
        self.neighs[depth] = neigh.view(-1, 27)  

      else:
        child_p = self.label[depth-1]
        mask = child_p >= 0
        neigh_p = self.neighs[depth-1][mask]   # (N, 27)
        neigh_p = neigh_p[:, self.parent_table]  # (N, 8, 27)
        child_p = child_p[neigh_p]  # (N, 8, 27)
        invalid = child_p < 0       # (N, 8, 27)
        neigh = child_p * 8 + self.child_table
        neigh[invalid] = -1
        self.neighs[depth] = neigh.view(-1, 27)


  def get_neigh(self,depth, occupied = False):
    """ Returns the neighborhoods given the depth at a filter 3*3*3.
    Args:
      depth (int): >0
      occupied (bool): If True, only returns the neighborhoods of the occupied octants.
    """
    neigh = self.neighs[depth]
    if occupied:
      l = self.label[depth]
      occupied_octant = l >= 0
      neigh = neigh[occupied_octant]
      valid = neigh >= 0
      neigh[valid] = l[neigh[valid]].long()  # remap the index

    return neigh

  ## here are four helpers
  def rng_grid(self, min, max):
    r''' Builds a mesh grid in :obj:`[min, max]` (:attr:`max` included).
    '''

    rng = torch.arange(min, max+1, dtype=torch.long, device=self.device)
    grid = torch.meshgrid(rng, rng, rng, indexing='ij')
    grid = torch.stack(grid, dim=-1).view(-1, 3)  # (27, 3)
    return grid

  def search_neigh_table(self):
    """construct the look up tables for neighborhood searching
    """
    center = self.rng_grid(2, 3)    # (8, 3)
    displacement = self.rng_grid(-1, 1)  # (27, 3)
    neigh = center.unsqueeze(1) + displacement  # (8, 27, 3)
    parent = torch.div(neigh, 2, rounding_mode='trunc')
    child = neigh % 2
    return torch.sum(parent * torch.tensor([9, 3, 1]).to(self.device), dim=2).to(self.device),torch.sum(child * torch.tensor([4, 2, 1]).to(self.device), dim=2).to(self.device)

  def scatter_addition(self, src, index, dim = -1):
    """ Writes all values from the tensor src into self at the indices specified in the index tensor. 
        modified from https://pytorch.org/docs/1.10/generated/torch.Tensor.scatter_.html#torch-tensor-scatter in a boardcasting fashion.
        when reduce="add"
    Args:
      src (torch.Tensor):  the source element(s) to scatter.
      index (torch.Tensor): the indices of elements to scatter, can be either empty or of the same dimensionality as src. 
                  When empty, the operation returns self unchanged.
      dim (int): The axis along which to index, default to -1.
      output(torch.Tensor):The destination tensor.
    """
    #broadcast first
    d,s,ind=dim,src,index

    if d < 0:
      d = s.dim() + d
    if ind.dim() == 1:
      for _ in range(0, d):
        ind = ind.unsqueeze(0)
    for _ in range(ind.dim(), s.dim()):
      ind = ind.unsqueeze(-1)

    index = ind.expand_as(s)
    #scatter_add then
    size = list(src.size())
    if index.numel() == 0:
      size[dim] = 0
    else:
      size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype).to(self.device)
    return out.scatter_add_(dim, index.to(self.device), src.to(self.device)).to(self.device)



  def build_ocforest(self, octrees):
    """ merge octrees into one batch is called ocforest here.
      this function is to build ocforest based on one batch of octrees.
    Args:
      octrees ([Octree]): one batch of octrees
    """ 
    # merge all tree into a forest
    ocforest=Octree(depth=octrees[0].depth,full_depth=octrees[0].full_depth,batch_size=len(octrees), device=octrees[0].device)
    # num and nempty_num
    num = torch.stack([octrees[i].num for i in range(ocforest.batch_size)], dim=1)

    nempty_num = torch.stack([octrees[i].nempty_num for i in range(ocforest.batch_size)], dim=1)

    ocforest.num=torch.sum(num,dim=1)
    ocforest.nempty_num=torch.sum(nempty_num,dim=1)

    nnum_cum = self.cumsum(nempty_num, dim=1)
    pad = torch.zeros_like(octrees[0].num).unsqueeze(1)
    nnum_cum = torch.cat([pad, nnum_cum], dim=1)
    
    # merge octre properties
    for d in range(ocforest.depth+1):
      #shuffle_keys
      keys = [None] * ocforest.batch_size
      for i in range(ocforest.batch_size):
        key = octrees[i].shuffle_keys[d] & ((1 << 48) - 1)  # clear the highest bits
        keys[i] = key | (i << 48)
      ocforest.shuffle_keys[d] = torch.cat(keys, dim=0)
      
      #labels
      children = [None] * ocforest.batch_size
      for i in range(ocforest.batch_size):
        child = octrees[i].label[d]
        mask = child >= 0
        child[mask] = child[mask] + nnum_cum[d, i]
        children[i] = child
      ocforest.label[d] = torch.cat(children, dim=0)

      # features
      if d == ocforest.depth:
        features = [octrees[i].features[d] for i in range(ocforest.batch_size)]
        ocforest.features[d] = torch.cat(features, dim=0)

      # normals
        normals = [octrees[i].normals[d] for i in range(ocforest.batch_size)]
        ocforest.normals[d] = torch.cat(normals, dim=0)

      # points
        points = [octrees[i].points[d] for i in range(ocforest.batch_size)]
        ocforest.points[d] = torch.cat(points, dim=0)

    return ocforest

  def cumsum(self,data, dim=1, exclusive=False):
    """ Extends :func:`torch.cumsum`
    Args:
      data (torch.Tensor): The input data.
      dim (int): The dimension to do the operation over.default to 1.
      exclusive (bool): If false, the behavior is the same as torch.cumsum.
          if true, returns the cumulative sum exclusively. Note that if ture,
          the shape of output tensor is larger by 1 than data in the
          dimension where the computation occurs.
    """

    out = torch.cumsum(data, dim)

    if exclusive:
      size = list(data.size())
      size[dim] = 1
      zeros = data.new_zeros(size)
      out = torch.cat([zeros, out], dim)
    return out
  
  def meshgrid(self, min, max):
    """ create a mesh grid in [min, max]
    """
    rng = torch.arange(min, max+1, dtype=torch.long, device=self.device)
    grid = torch.meshgrid(rng, rng, rng, indexing='ij')
    grid = torch.stack(grid, dim=-1).view(-1, 3)  
    return grid

