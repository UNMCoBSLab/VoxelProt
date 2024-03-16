# -*- coding: utf-8 -*-
"""maxpooling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14MApiOVAPrhQhUnD-yzKP7U5nGe6yxbX
"""

import torch
import torch.nn
class MaxPool(torch.nn.Module):
  """this function is to create the max pooling layer
  Args:
    only_occupied (bool): If false, gets the input signal of all octants in the finest level.
              including the non-occupied octants, whose input signal vector will set to [0.0,0.0,0.0,0.0,0.0,0.0] 
              default to False 

  """
  def __init__(self, only_occupied=False):
    super().__init__()
    self.only_occupied = only_occupied

  def forward(self, input_signal, octree, depth):
    """Args:
      input_signal(torch.Tensor)The input tensor.
      octree(Octree):The corresponding octree.
      depth(int):The depth of current octree. After pooling, the corresponding
        depth decreased by 1.

    """
    if self.only_occupied:
      input_signal = self.padding(input_signal, octree, depth, padding_val=float('-inf'))
    
    input_signal = input_signal.view(-1, 8, input_signal.shape[1])
    out,_ = input_signal.max(dim=1)
    
    if not self.only_occupied:
     out = self.padding(out, octree, depth-1)
    return out

  # 1 helper method
  def padding(self, inputsignal, octree, depth, pading_val=0.0):
    """ Pads pading_val to assign non-occupied octant a vector.
    Args:
      input_signal (torch.Tensor): The input tensor (num_occupied,6)
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree,default to the finest level.
      pading_val (float): The padding value. default to 0.0
    """
    mask = octree.label[depth]>= 0
    size = (octree.num[depth], inputsignal.shape[1])  
    fill_value=pading_val
    out = torch.full(size, fill_value, dtype=inputsignal.dtype, device=inputsignal.device)
    out[mask] = inputsignal
    return out