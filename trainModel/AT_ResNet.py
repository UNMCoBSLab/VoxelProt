import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os,random,csv,sys
from VoxelProt.trainModel.pickleoctree import *
from VoxelProt.trainModel.octree import Octree
from VoxelProt.trainModel.inputsignal import InputSignal
from VoxelProt.trainModel.maxpooling import MaxPool
from VoxelProt.trainModel.globalAvgPool import GlobalAvgPool
from VoxelProt.trainModel.convolutionallayer import *

#this training network includes the energy
#this add q,k,more atomes layers
"""     q_cg=self.cg_q(D_cg)
        k_cg=self.cg_k(D_cg)   
        k_v=self.v_k(D_v) 
        alpha11=torch.mul(k_cg, q_cg)
        alpha12=torch.mul(k_v, q_cg) 
        D=torch.mul(alpha11,D_cg)+torch.mul(alpha12,D_v)
"""

class ProjectionResBlock(nn.Module):
    def __init__(self,channel_in,channel_out,stride=1,only_occupied=False,momentum=0.01,eps=0.001,max_buffer=int(2e8)):
        super().__init__()
        self.expansion = 4
        self.channel_in    = channel_in
        self.channel_out    = channel_out
        self.intermediate_channels   = int(self.channel_out/self.expansion)
        self.stride        = stride
        self.only_occupied = only_occupied
        self.momentum      = momentum
        self.eps           = eps
        self.max_buffer    = max_buffer

        #the first 1×1 conv uses Conv1×1 + BN + ReLU
        self.conv_bn_relu1 = ConvBnRelu(self.channel_in,self.intermediate_channels,kernel=1,stride=self.stride,only_occupied=self.only_occupied,
            momentum=self.momentum,eps=self.eps,max_buffer=self.max_buffer)

        #the 3×3 conv uses Conv3×3 + BN + ReLU
        self.conv_bn_relu2 = ConvBnRelu(self.intermediate_channels,self.intermediate_channels, kernel=3,stride=self.stride,only_occupied=self.only_occupied,
            momentum=self.momentum,eps=self.eps,max_buffer=self.max_buffer)
        
        #the 1×1 conv uses Conv1×1 + BN 
        self.conv3 = OctreeConv(self.intermediate_channels, self.channel_out, kernel=1, stride=self.stride, only_occupied=self.only_occupied,max_buffer=self.max_buffer)
        self.bn3 = nn.BatchNorm1d(self.channel_out, self.eps, self.momentum)
        
        #the projection
        self.conv_projection = OctreeConv(self.channel_in, self.channel_out, kernel=1, stride=self.stride, only_occupied=self.only_occupied,max_buffer=self.max_buffer)
        self.bn_projection = nn.BatchNorm1d(self.channel_out, self.eps, self.momentum)
        #add together
        self.relu = nn.ReLU(inplace=True)  

            
    def forward(self,x,octree,depth):
        projection = x.clone()        
        #the first 1×1 conv uses Conv1×1 + BN + ReLU
        x = self.conv_bn_relu1(x,octree,depth)
        
        #the 3×3 conv uses Conv3×3 + BN + ReLU
        x = self.conv_bn_relu2(x,octree,depth)
        
        #the 1×1 conv uses Conv1×1 + BN 
        x = self.conv3(x,octree,depth)
        x = self.bn3(x)

        #projection
        p = self.conv_projection(projection,octree,depth)
        p = self.bn_projection(p)
        #elementwise add, 
        x =x + p
        
        #then final ReLU    
        x = self.relu(x)         
        return x

        
class IdentityResBlock(nn.Module):
    def __init__(self,channel_in,stride=1,only_occupied=False,momentum=0.01,eps=0.001,max_buffer=int(2e8)):
        super().__init__()
        self.expansion = 4
        self.channel_in    = channel_in
        self.intermediate_channels   = int(self.channel_in/self.expansion)
        self.stride        = stride
        self.only_occupied = only_occupied
        self.momentum      = momentum
        self.eps           = eps
        self.max_buffer    = max_buffer
        #the first 1×1 conv uses Conv1×1 + BN + ReLU
        self.conv_bn_relu1 = ConvBnRelu(self.channel_in,self.intermediate_channels,kernel=1,stride=self.stride,only_occupied=self.only_occupied,
            momentum=self.momentum,eps=self.eps,max_buffer=self.max_buffer)

        #the 3×3 conv uses Conv3×3 + BN + ReLU
        self.conv_bn_relu2 = ConvBnRelu(self.intermediate_channels,self.intermediate_channels, kernel=3,stride=self.stride,only_occupied=self.only_occupied,
            momentum=self.momentum,eps=self.eps,max_buffer=self.max_buffer)
        
        #the 1×1 conv uses Conv1×1 + BN 
        self.conv3 = OctreeConv(self.intermediate_channels, self.channel_in, kernel=1, stride=self.stride, only_occupied=self.only_occupied,max_buffer=self.max_buffer)
        self.bn = nn.BatchNorm1d(self.channel_in, self.eps, self.momentum)
        self.relu = nn.ReLU(inplace=True)  

            
    def forward(self,x,octree,depth):
        identity = x.clone()        
        #the first 1×1 conv uses Conv1×1 + BN + ReLU
        x = self.conv_bn_relu1(x,octree,depth)
        
        #the 3×3 conv uses Conv3×3 + BN + ReLU
        x = self.conv_bn_relu2(x,octree,depth)
        
        #the 1×1 conv uses Conv1×1 + BN 
        x = self.conv3(x,octree,depth)
        x = self.bn(x)
        
        #elementwise add, 
        x += identity
        
        #then final ReLU    
        x = self.relu(x)         
        return x
                 
class AT_ResNet(nn.Module):
    def __init__(self,
                 channel_in=16,
                 num_classes=2,
                 kernel=3,
                 depth=4,
                 stride=1,
                 only_occupied=False,
                 momentum=0.01,
                 eps=0.001,
                 max_buffer=int(2e8)):
        super().__init__()
        self.kernel        = kernel
        self.depth         = depth
        self.channel_in    = channel_in
        self.num_classes   = num_classes
        self.channel_out   = [2**(max(11 - i, 3)) for i in range(self.depth, 2, -1)]
        self.stride        = stride
        self.only_occupied = only_occupied
        self.momentum      = momentum
        self.eps           = eps
        self.max_buffer    = max_buffer

        #------------------------------  chem+geo module
        self.input_signal = InputSignal(self.only_occupied)
    
        # Layer 1: conv → batchnorm → relu
        self.conv_bn_relu1 = ConvBnRelu(channel_in,self.channel_out[0],kernel=1,stride=self.stride,only_occupied=self.only_occupied,
            momentum=self.momentum,eps=self.eps,max_buffer=self.max_buffer)


        #Stage 2:3 identity ResBlocks
        self.iRB1 = IdentityResBlock(self.channel_out[0],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB2 = IdentityResBlock(self.channel_out[0],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB3 = IdentityResBlock(self.channel_out[0],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))        
        self.mp = MaxPool()

        # Stage 2:1 projection ResBlocks + 2 identity ResBlocks
        self.pRB1 = ProjectionResBlock(self.channel_out[0],self.channel_out[1],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB5 = IdentityResBlock(self.channel_out[1],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB6 = IdentityResBlock(self.channel_out[1],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8)) 

        # Stage 3: 1 8x8x8x averageMaxPooling
        self.ave_mp = GlobalAvgPool()



        #------------------------------ energy module
        self.input_signal_e = InputSignal(self.only_occupied)
    
        # Layer 1: conv → batchnorm → relu
        self.conv_bn_relu1_e = ConvBnRelu(2,self.channel_out[0],kernel=1,stride=self.stride,only_occupied=self.only_occupied,
            momentum=self.momentum,eps=self.eps,max_buffer=self.max_buffer)


        #Stage 2:3 identity ResBlocks
        self.iRB1_e= IdentityResBlock(self.channel_out[0],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB2_e = IdentityResBlock(self.channel_out[0],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB3_e = IdentityResBlock(self.channel_out[0],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))        
        self.mp_e = MaxPool()

        # Stage 2:1 projection ResBlocks + 2 identity ResBlocks
        self.pRB1_e = ProjectionResBlock(self.channel_out[0],self.channel_out[1],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB5_e = IdentityResBlock(self.channel_out[1],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8))
        self.iRB6_e = IdentityResBlock(self.channel_out[1],stride=1,only_occupied=False,momentum=0.01,eps=0.001, max_buffer=int(2e8)) 

        # Stage 3: 1 8x8x8x averageMaxPooling
        self.ave_mp_e = GlobalAvgPool()

        # Stage 4: Create the input from chemical and geo
        self.cg_q=nn.Linear(self.channel_out[-1],self.channel_out[-1])
        #create the input from energy
        #self.v_q=nn.Linear(self.channel_out[-1],self.channel_out[-1])
        #create the input from chemical and geo
        self.cg_k=nn.Linear(self.channel_out[-1],self.channel_out[-1])
        #create the input from energy
        self.v_k=nn.Linear(self.channel_out[-1],self.channel_out[-1])
        self.fcs=nn.Linear(self.channel_out[-1],self.num_classes)
        
        
    def forward(self, octree_cg,octree_v, batch_size):
        #--------------------------cg module
        # Build the input feature tensor: shape (N, channel_in)--cg
        x_cg = self.input_signal(octree_cg).to(torch.float)

        # Layer 1: conv1--cg
        x_cg = self.conv_bn_relu1(x_cg, octree_cg, octree_cg.depth)
        # Stage 2:3 identity ResBlocks
        x_cg = self.iRB1(x_cg, octree_cg, octree_cg.depth)
        x_cg = self.iRB2(x_cg, octree_cg, octree_cg.depth)
        x_cg = self.iRB3(x_cg, octree_cg, octree_cg.depth)
        x_cg = self.mp(x_cg, octree_cg, octree_cg.depth)

        
        # Stage 2:1 projection ResBlocks + 2 identity ResBlocks--cg
        x_cg = self.pRB1(x_cg, octree_cg, octree_cg.depth-1)
        x_cg = self.iRB5(x_cg, octree_cg, octree_cg.depth-1)
        x_cg = self.iRB6(x_cg, octree_cg, octree_cg.depth-1)

        # Stage 3: 1 8x8x8x averageMaxPooling--cg
        x_cg = self.ave_mp(x_cg, octree_cg,depth = 2)
        
        #--------------------------v module      
        # Build the input feature tensor: shape (N, channel_in)--v
        x_v = self.input_signal_e(octree_v).to(torch.float)

        # Layer 1: conv1--v
        x_v = self.conv_bn_relu1_e(x_v, octree_v, octree_v.depth)
        # Stage 2:3 identity ResBlocks
        x_v = self.iRB1_e(x_v, octree_v, octree_v.depth)
        x_v = self.iRB2_e(x_v, octree_v, octree_v.depth)
        x_v = self.iRB3_e(x_v, octree_v, octree_v.depth)
        x_v = self.mp_e(x_v, octree_v, octree_v.depth)

        
        # Stage 2:1 projection ResBlocks + 2 identity ResBlocks--v
        x_v = self.pRB1_e(x_v, octree_v, octree_v.depth-1)
        x_v = self.iRB5_e(x_v, octree_v, octree_v.depth-1)
        x_v = self.iRB6_e(x_v, octree_v, octree_v.depth-1)

        # Stage 3: 1 8x8x8x averageMaxPooling--cg
        x_v = self.ave_mp_e(x_v, octree_v,depth = 2)
        
        #--------------------------selfattention      
        q_cg=self.cg_q(x_cg)
        #q_v=self.v_q(x_v)
        k_cg=self.cg_k(x_cg)   
        k_v=self.v_k(x_v) 
        alpha11=torch.mul(k_cg, q_cg)
        alpha12=torch.mul(k_v, q_cg) 
        D=torch.mul(alpha11,x_cg)+torch.mul(alpha12,x_v)
        #flatten  
        xx=torch.nn.Flatten()
        D=xx(D)

        if batch_size==1:
          D = D[None, :]    
        else:
          D=torch.stack(torch.split(D,batch_size),dim=1)

        D=D.reshape(D.shape[0],-1)        
        # Stage 4: fc
        D=self.fcs(D)
        return D
