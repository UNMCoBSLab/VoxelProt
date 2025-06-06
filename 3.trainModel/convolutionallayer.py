import math
import torch
import torch.nn as nn
from torch.autograd import Function

class OctreeConvFunction(Function):
  """ We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

    data(torch.Tensor):
    weights(torch.Tensor):
    octree(Octree):
    depth(int):
    in_channels(int):
    out_channels(int)
    stride(int)
    only_occupied(bool)
    max_buffer(int)
  """

  @staticmethod
  def forward(ctx, inputsignal, weights, octree, depth, in_channels, out_channels,stride=1,only_occupied = False, max_buffer=int(2e8)):
    octree_conv = OctreeConv(in_channels, out_channels, stride, only_occupied, max_buffer)
    octree_conv.setup(octree, depth)
    out = octree_conv.check_and_init(inputsignal)
    out = octree_conv.forward_gemm(out, inputsignal, weights)

    ctx.save_for_backward(inputsignal, weights)
    ctx.octree_conv = octree_conv
    return out

  @staticmethod
  def backward(ctx, grad):
    inputsignal, weights = ctx.saved_tensors
    octree_conv = ctx.octree_conv

    grad_out = None
    if ctx.needs_input_grad[0]:
      grad_out = torch.zeros_like(inputsignal)
      grad_out = octree_conv.backward_gemm(grad_out, grad, weights)

    grad_w = None
    if ctx.needs_input_grad[1]:
      grad_w = torch.zeros_like(weights)
      grad_w = octree_conv.weight_gemm(grad_w, inputsignal, grad)

    return (grad_out, grad_w) + (None,) * 8


# create the octree_conv_function
octree_conv = OctreeConvFunction.apply



class OctreeConv(torch.nn.Module):
  """ octree convolution layer. no bias term right now.
  Args:
      channel_in (int): Number of input channels.
      channel_out (int): Number of output channels.
      stride (int): The stride of the convolution (1 or 2 and default to 1).
      only_occupied (bool): If false, gets the input signal of all octants in the finest level.
              including the non-occupied octants, whose input signal vector will set to [0.0,0.0,0.0,0.0,0.0,0.0] 
              default to False 
      max_buffer (int): The maximum number of elements in the buffer.     
  """
  def __init__(self,in_channels,out_channels, stride=1, only_occupied=False, max_buffer = int(2e8)):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    self.only_occupied = only_occupied
    self.max_buffer = max_buffer

    #parameters about filter
    self.filter_size = "3*3*3"
    self.dim_filter = 27

    #Initialize convolution weights
    self.weights_shape = (self.dim_filter, self.in_channels, self.out_channels)
    self.weights = torch.nn.Parameter(torch.Tensor(*self.weights_shape))
    self.Initialize()

  def Initialize(self):
    """ Initialize convolution weights. https://pytorch.org/docs/stable/nn.init.html
    """

    std = math.sqrt(2.0 / float((self.weights_shape[0] * self.weights_shape[1]) + (self.weights_shape[0] * self.weights_shape[2])))
    with torch.no_grad():
      return self.weights.uniform_(-1*(math.sqrt(3.0) * std), (math.sqrt(3.0) * std))

  def forward(self, inputSignal, octree, depth):
    """
      Args:
        inputSignaldata (torch.Tensor): The input data.
        octree (Octree): The corresponding octree.
        depth (int): The depth of current octree.
    """
    D = octree_conv(inputSignal, self.weights, octree, depth, self.in_channels, self.out_channels, self.stride, self.only_occupied, self.max_buffer)
    if self.stride == 2 and not self.only_occupied:
      D = self.padding(D, octree, depth-1)
    return D


  def setup(self, octree, depth):
    """ Setup the shapes of each buffer. Be called before forward_gemm, backward_gemm and weight_gemm.
    Args:
       octree(Octree)
       depth(int):The depth of tensors
    """

    # The in_depth and out_depth are the octree depth of the input and output data;
    self.in_depth = depth
    self.out_depth = depth
    # neigh_depth is the octree depth of the neighborhood information
    self.neigh_depth = depth

    if self.stride == 2:
      self.out_depth = depth - 1

    # The height of tensors
    if self.only_occupied:
      self.in_h = octree.nempty_num[self.in_depth]
      self.out_h = octree.nempty_num[self.out_depth]
    else:
      self.in_h = octree.num[self.in_depth]
      self.out_h = octree.num[self.out_depth]
      if self.stride == 2:
        self.out_h = octree.nempty_num[self.out_depth]

    self.in_shape = (self.in_h, self.in_channels)
    self.out_shape = (self.out_h, self.out_channels)

    # The neighborhood indices
    self.neigh = octree.get_neigh(self.neigh_depth, self.only_occupied)

    # The heigh and number of the temporary buffer
    self.buffer_n = 1
    self.buffer_h = self.neigh.shape[0]
    ideal_size = self.buffer_h * self.neigh.shape[1] * self.in_channels
    if ideal_size > self.max_buffer:
      self.buffer_n = (ideal_size + self.max_buffer - 1) // self.max_buffer
      self.buffer_h = (self.buffer_h + self.buffer_n - 1) // self.buffer_n
    self.buffer_shape = (self.buffer_h, 27, self.in_channels)

  def check_and_init(self, inputsignal):
    """ Checks the input data and initializes the shape of output data.
      inputsignal(torch.Tensor)
    """

    # Check the shape of input data
    check = tuple(inputsignal.shape) == self.in_shape
    torch._assert(check, 'The shape of input data is wrong.')

    # Init the output data
    out = inputsignal.new_zeros(self.out_shape)
    return out

  def forward_gemm(self, out, inputsignal, weights):
    """ Peforms the forward pass of octree-based convolution.
      Args:
        out(torch.Tensor)
        inputsignal(torch.Tensor)
        weights(torch.Tensor)
    """

    # Initialize the buffer
    buffer = inputsignal.new_empty(self.buffer_shape)

    # Loop over each sub-matrix
    for i in range(self.buffer_n):
      start = i * self.buffer_h
      end = (i + 1) * self.buffer_h

      # The boundary case in the last iteration
      if end > self.neigh.shape[0]:
        dis = end - self.neigh.shape[0]
        end = self.neigh.shape[0]
        buffer, _ = buffer.split([self.buffer_h-dis, dis])

      # Perform octree2col
      neigh_i = self.neigh[start:end]
      valid = neigh_i >= 0
      buffer.fill_(0)
      buffer[valid] = inputsignal[neigh_i[valid]]

      # The sub-matrix gemm
      out[start:end] = torch.mm(buffer.flatten(1, 2).to(torch.float), weights.flatten(0, 1).to(torch.float))

    return out

  def backward_gemm(self,grad_out,grad,weights):
    """ Performs the backward pass of octree-based convolution. 
       grad_out(tensor):
       grad(tensor):
       weights(tensor):
    """

    # Loop over each sub-matrix
    for i in range(self.buffer_n):
      start = i * self.buffer_h
      end = (i + 1) * self.buffer_h

      # The boundary case in the last iteration
      if end > self.neigh.shape[0]:
        end = self.neigh.shape[0]

      # The sub-matrix gemm
      buffer = torch.mm(grad[start:end], weights.flatten(0, 1).t())
      buffer = buffer.view(-1, self.buffer_shape[1], self.buffer_shape[2])

      # Performs col2octree
      neigh_i = self.neigh[start:end]
      valid = neigh_i >= 0
      grad_out = self.scatter_addition(buffer[valid], neigh_i[valid], dim=0, out=grad_out)

    return grad_out

  def weight_gemm(self, grad_w, inputsignal, grad):
    """ Computes the gradient of the weight matrix.
      Args:
        grad_w(torch.Tensor)
        inputsignal(torch.Tensor):
        grad(torch.Tensor):
    """

    # Record the shape of grad_w
    grad_w_shape = grad_w.shape
    grad_w = grad_w.flatten(0, 1)

    # Initialize the buffer
    buffer = inputsignal.new_empty(self.buffer_shape)

    # Loop over each sub-matrix
    for i in range(self.buffer_n):
      start = i * self.buffer_h
      end = (i + 1) * self.buffer_h

      # The boundary case in the last iteration
      if end > self.neigh.shape[0]:
        d = end - self.neigh.shape[0]
        end = self.neigh.shape[0]
        buffer, _ = buffer.split([self.buffer_h-d, d])

      # Perform octree2col
      neigh_i = self.neigh[start:end]
      valid = neigh_i >= 0
      buffer.fill_(0)
      buffer[valid] = inputsignal[neigh_i[valid]]

      # Accumulate the gradient via gemm
      grad_w.addmm_(buffer.flatten(1, 2).t(), grad[start:end])

    return grad_w.view(grad_w_shape)

  
  # 3 helper methods
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



  def scatter_addition(self, src, index, out, dim = -1):
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

    return out.scatter_add_(dim, index, src)

  def extra_repr(self) -> str:
    """ Sets the extra representation of the module.
    https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
    """
    return ('in_channels={}, out_channels={}, kernel_size={}, stride={}, '
            'only_occupied={}').format(self.in_channels, self.out_channels,
             "3*3*3", self.stride, self.only_occupied) 
            

import torch
import torch.nn as nn
class ConvBnRelu(torch.nn.Module):
  """ A block contains Conv+BatchNorm+Relu.
    Args:
      channel_in (int): Number of input channels.
      channel_out (int): Number of output channels.
      stride (int): The stride of the convolution (1 or 2 and default to 1).
      only_occupied (bool): If false, gets the input signal of all octants in the finest level.
              including the non-occupied octants, whose input signal vector will set to [0.0,0.0,0.0,0.0,0.0,0.0] 
              default to False 
      momentum(float):the value used for the running_mean and running_var computation. 
              Can be set to None for cumulative moving average (i.e. simple average). Default: 0.01
      eps(float): a value added to the denominator for numerical stability. Default: 0.001
      max_buffer (int): The maximum number of elements in the buffer.

  """

  def __init__(self, channel_in, channel_out, stride=1, only_occupied=False, momentum=0.01, eps=0.001,max_buffer=int(2e8)):
    super().__init__()
    self.conv = OctreeConv(channel_in, channel_out, stride=stride, only_occupied=only_occupied,max_buffer=max_buffer)
    self.bn = nn.BatchNorm1d(channel_out, eps, momentum)
    self.relu = nn.ReLU(inplace=True)  

  def forward(self, inputSignal,octree,depth):
    """
     A sequence of Conv, BatchNorm, Relu
     inputSignal(torch.Tensor):
     octree(Octree):
     depth(int):
    """
    D = self.conv(inputSignal, octree, depth)
    D = self.bn(D)
    D = self.relu(D)
    return D
