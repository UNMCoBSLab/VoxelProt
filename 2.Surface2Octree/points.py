#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from math import pi
import math

class Points:
  """ Represents a point cloud and contains some elementary transformations.
  Args:
    points (torch.Tensor): The coordinates of the points with a shape of (N, 3).like #[x-cor, y-cor, z-cor]
    normals (torch.Tensor): The point normals with a shape of (N, 3).[normal_x, normal_y, normal_z]
    features (torch.Tensor): The point features with a shape of (N, 13).[hydropathy, electrostatics, hydrogenpotential,atomtype*10]
    labels (torch.Tensor): The point labels with a shape of (N, K).like if determine a binding site is ADP or not. Labels=torch.tensor([[0],.....,[1]]) 
    origin(torch.tensor): the center of box.like tensor([ 0.9505, -0.2921, -0.1059])
    length(int): the box length,default to 16

  """

  def __init__(self,origin, points, normals, features, labels, device, length=16):
    self.device = device
    self.origin=origin
    self.length=length
    self.points=points

    self.rootIndex=self.generateRoot(self.origin,self.length)
 
    self.all=torch.tensor(np.array([points[i].tolist()+normals[i].tolist()+features[i].tolist()+labels[i].tolist() for i in self.rootIndex])).to(dtype=torch.float64)

    self.points= self.all[:,0:3]
    self.normals =self.all[:,3:6]
    self.features = self.all[:,6:(features.shape[1]+6)]
    self.labels = self.all[:,(features.shape[1]+6):]
    self.points=self.normalize()

    

  def generateRoot(self,origin,boxlength):
    rv=[]
    for ind in range(self.points.size(0)):
      x,y,z=self.points[ind,0:3].detach().numpy()
      if abs(origin[0]-x)<boxlength*0.5:
        if abs(origin[1]-y)<boxlength*0.5:
          if abs(origin[2]-z)<boxlength*0.5:
            rv.append(ind)
    return rv
  
  def normalize(self, scale = 1.0):
    """ Normalizes the point cloud to [-scale, scale].
      Args:
        scale (float): The scale factor.default to 1.0
    """
    min_v=-1*scale
    max_v=scale
    return torch.tensor(np.interp(self.points,(self.points.min(), self.points.max()), (min_v, max_v)))

  def rotate(self,angle,axis):
    """this function is to rotate the Point
    Args:
      angle: the rotate angle, in Degrees, like 90
      axis(char):x,y or z. rotating the points along x,y,or z axis
    """
    angle=math.radians(angle)
    cos,sin=math.cos(angle),math.sin(angle)

    if axis=="x":
      rot= torch.Tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]]).to(dtype=torch.float64)
    elif axis=="y":
      rot= torch.Tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]).to(dtype=torch.float64)
    else:
      rot= torch.Tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]).to(dtype=torch.float64)
    rot = rot.to(self.device)
    #rotate the coordinates of points
    self.points = self.points.to(self.device) @ rot
    #rotate the coordinates of normals
    self.normals = self.normals.to(self.device) @ rot

