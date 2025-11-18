import torch
import torch.nn as nn
class GlobalAvgPool(nn.Module):
    """ to global average pooling
    Args:
        only_occupied (bool): If True, input_signal contains only occupied nodes and will be
                              padded to the full grid before averaging.
                              If False, input_signal is already the full grid at depth 2.
    """
    def __init__(self, only_occupied: bool = False):
        super().__init__()
        self.only_occupied = only_occupied

    def forward(self, input_signal , octree, depth = 2) :
        """
        Args:
          input_signal (Tensor): shape (nempty_num[2], C) if only_occupied=True, else (num[2], C)
          octree (Octree):      your octree instance
          depth (int):          must be 2 for this layer
        Returns:
          Tensor of shape (1, C): the global average–pooled features.
        """
        x = input_signal
        if self.only_occupied:
            x = self.padding(x, octree, depth, padding_val=0.0)

        # Average over all nodes at depth 2
        out = x.mean(dim=0, keepdim=True)
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
