#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pykeops
from pykeops.torch import LazyTensor
import torch.nn.functional as F
import numpy
import csv
import pandas as pd

def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
    
    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
  """Encodes the block-diagonal structure associated to a batch vector."""
  if batch_x is None and batch_y is None:
    return None  # No batch processing
  elif batch_y is None:
    batch_y = batch_x  # "symmetric" case
  ranges_x, slices_x = ranges_slices(batch_x)
  ranges_y, slices_y = ranges_slices(batch_y)
  return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x

def cross(vec1,vec2):
  """Computes cross product of two vectors.
    
    Args:
        vec1: vector 1.
        vec2: vector 2 .
    Returns:
        Tensor: (tensor)the cross product of two vectors 
  """       
  x,y,z=vec1[0],vec1[1],vec1[2]
  u,v,w=vec2[0],vec2[1],vec2[2]
  return torch.tensor([y*w- z*v,z*u - x*w,x*v- y*u ])



def dot(vec1,vec2):
  """Computes dot product of two vectors.
    
    Args:
        vec1: vector 1.
        vec2: vector 2 .
    Returns:the cross product of two vectors 
  """
  x,y,z=vec1[0],vec1[1],vec1[2]
  u,v,w=vec2[0],vec2[1],vec2[2]
  return x*u + y*v + z*w


#evaluation points:
#vote 

def evaluation_points2(model,dataset,label,device="cuda"):

    yes=0
    no=0
    
    for each in dataset:
            inputs=readOctree(each)
            labels=torch.tensor([label]).to(device) 
            outputs = model(inputs,batch_size=labels.shape[0])
            #log_softmax = F.log_softmax(outputs,dim=1)
            #loss = torch.nn.functional.nll_loss(log_softmax, labels)
            pred = torch.argmax(outputs, dim=1)
            if (pred==0).item():
                no=no+1
            else:
                yes=yes+1
            
    if yes>=no:
        pred=torch.tensor([1]).to(device)
    else:
        pred=torch.tensor([0]).to(device) 
    
    return pred.eq(torch.tensor([label]).to(device)).float().mean()


def write_to_txt(lst,file_name):
    file = open(file_name,'w')
    for each in lst:
        file.write(str(each)+"\n")
    file.close()
    

def feathers2CSV(features,address,name):
    x_np=features.numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv(address+name+'.csv',index=False,header=False)     

