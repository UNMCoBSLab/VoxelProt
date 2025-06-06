import torch
"""
This module is to create a shuffle key look up table
So if the input is a (x,y,z) coordinate, the output is its corresponding shuffle key
and if the input is a shuffle key, the output is its x,y,z range
"""
class KeyTable:
  def __init__(self,depth=4):
    num_256 = torch.arange(256, dtype=torch.int64)
    num_512 = torch.arange(512, dtype=torch.int64)
    zero_256 = torch.zeros(256, dtype=torch.int64)
    self.depth=depth
    self._encode = {torch.device('cpu'): (self.xyz2shuffleKey(num_256, zero_256, zero_256,self.depth), self.xyz2shuffleKey(zero_256, num_256, zero_256, self.depth), self.xyz2shuffleKey(zero_256, zero_256, num_256,self.depth))}
    self._decode = {torch.device('cpu'): self.shuffleKey2xyz(num_512, self.depth+1)}

  def encode_table(self, device=torch.device('cpu')):
    if device not in self._encode:
      self._encode[device] = tuple(each.to(device) for each in self._encode[torch.device('cpu')])
    return self._encode[device]

  def decode_table(self, device=torch.device('cpu')):
    if device not in self._decode:
      self._decode[device] = tuple(each.to(device) for each in self._decode[torch.device('cpu')])
    return self._decode[device]

  def xyz2shuffleKey(self, x, y, z, depth):
    shuffleKey = torch.zeros_like(x)
    for i in range(depth):
      mask = 1 << i
      shuffleKey = (shuffleKey | ((x & mask) << (2 * i + 2)) |((y & mask) << (2 * i + 1)) |((z & mask) << (2 * i + 0)))
    return shuffleKey

  def shuffleKey2xyz(self, shuffleKey, depth):
    x,y,z = torch.zeros_like(shuffleKey),torch.zeros_like(shuffleKey),torch.zeros_like(shuffleKey)
    for i in range(depth):
      x = x | ((shuffleKey & (1 << (3 * i + 2))) >> (2 * i + 2))
      y = y | ((shuffleKey & (1 << (3 * i + 1))) >> (2 * i + 1))
      z = z | ((shuffleKey & (1 << (3 * i + 0))) >> (2 * i + 0))
    return x, y, z

##pre-generate a shuffle key look up table with depth=4
_shuffleKeyTable = KeyTable(4)


def xyz2shufflekey(x, y, z, depth=4):
  """Encodes (x,y,z) to its corresponding shuffle key
  based on  the pre-generate shuffle key tables. 
  Args:
    x,y,z (torch.Tensor): The x,y,z coordinate.
    depth (int): The depth of the shuffled key, default to 4 and < 9.
  """
  Encode_X, Encode_Y, Encode_Z = _shuffleKeyTable.encode_table(x.device)
  # Rounding x,y,z 
  x, y, z = x.long(), y.long(), z.long()
  mask = (1 << depth) - 1
  key = Encode_X[x & mask] | Encode_Y[y & mask] | Encode_Z[z & mask]
  return key


def shufflekey2xyz(shuffleKey, depth=4):
  """Decodes the shuffled key to its corresponding (x,y,z) 
  based on  the pre-generate shuffle key tables.
  Args:
    key (torch.Tensor): The shuffle key.
    depth (int): The depth of the shuffled key, default to 4 and < 9.
  """

  Decode_X, Decode_Y, Decode_Z = _shuffleKeyTable.decode_table(shuffleKey.device)
  x, y, z = torch.zeros_like(shuffleKey), torch.zeros_like(shuffleKey), torch.zeros_like(shuffleKey)

  b = shuffleKey >> 48
  shuffleKey = shuffleKey & ((1 << 48) - 1)

  n = (depth + 2) // 3
  for i in range(n):
    k = shuffleKey >> (i * 9) & 511
    x = x | (Decode_X[k] << (i * 3))
    y = y | (Decode_Y[k] << (i * 3))
    z = z | (Decode_Z[k] << (i * 3))

  return x, y, z, b
