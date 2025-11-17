import torch
import torch.nn as nn
from VoxelProt.trainModel.octree import Octree
from VoxelProt.trainModel.inputsignal import InputSignal
from VoxelProt.trainModel.maxpooling import MaxPool
from VoxelProt.trainModel.convolutionallayer import *
class VGG_net(nn.Module):
  """
    Octree-based VGG_net for classification
    Args:
      channel_in (int): Number of input channels. default to 6.
      num_classes (int): number of classes.default to 2
      depth(int): the depth of octree,default to 4
      stride (int): The stride of the convolution (1 or 2 and default to 1).
      only_occupied (bool): If false, gets the input signal of all octants in the finest level.
              including the non-occupied octants, whose input signal vector will set to [0.0,0.0,0.0,0.0,0.0,0.0] 
              default to False    

      eps(float): a value added to the denominator for numerical stability. Default: 0.001
      momentum(float):the value used for the running_mean and running_var computation. 
              Can be set to None for cumulative moving average (i.e. simple average). Default: 0.01
      max_buffer (int): The maximum number of elements in the buffer.
  """
  def __init__(self,channel_in=16,num_classes=2,depth=4,stride=1,only_occupied=False,momentum=0.01,eps=0.001,max_buffer = int(2e8)):
    super(VGG_net,self).__init__()

    self.depth=depth
    self.channel_in=channel_in
    self.num_classes=num_classes
    self.channel_out = [2**(max(11-i,3)) for i in range(self.depth,2,-1)]
    self.momentum=momentum
    self.only_occupied=only_occupied
    self.eps=eps
    self.stride=stride
    self.max_buffer= max_buffer 

    # get the input signal vector
    self.input_signal = InputSignal(self.only_occupied)

    # create the convolution layers
    self.conv_layers=self.create_conv_layers(self.channel_out)
 
    # Then flatten and create the Linear Layers
    
    self.fcs=nn.Sequential(  
        nn.Linear(self.channel_out[-1]*4*4*4,512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512,self.num_classes),
     )


  def forward(self,octree,batch_size):
    """forward function
    Args:
      octree(Octree):
    """
    depth=octree.depth
    D=self.input_signal(octree)

    for i in range(depth-2):
      d=depth-i
      D=self.conv_layers[2*i](D.to(torch.float),octree,d)
      D=self.conv_layers[2*i+1](D.to(torch.float),octree,d)

    #flatten  
    xx=torch.nn.Flatten()
    D=xx(D)

    if batch_size==1:
      D = D[None, :]    
    else:
      D=torch.stack(torch.split(D,batch_size),dim=1)

    D=D.reshape(D.shape[0],-1)
    D=self.fcs(D)
    return D

  def create_conv_layers(self,architecture):
    """to create the convulutional layers
    Args:
      architecture(list): the architecture of nn. like [128,256,512]
    """
    layers=[]
    c_in=self.channel_in

    for each in architecture:
      c_out=each
      layers +=[ConvBnRelu(c_in, c_out, stride=self.stride,only_occupied=self.only_occupied, momentum=self.momentum, eps=self.eps,max_buffer = self.max_buffer)]
      c_in=c_out
      
      layers +=[MaxPool()]
    
    return nn.Sequential(*layers)

