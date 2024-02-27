#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#this contains  channels C,N,O,H
import torch
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors,KDTree
from vdwEnergy import *
class VDWPoints:
    def __init__(self,protein_atoms,atoms_tree,origin, device="cuda",length=16):
        self.psedoAtom=self.generatePsedoAtom(origin,atoms_tree,length)
        self.vdws=self.calcVDW(protein_atoms,atoms_tree,self.psedoAtom)
        self.device = device
        self.psedoAtom=self.normalize()
        
        
    def generatePsedoAtom(self,origin,atoms_tree,length):
        x0, y0, z0=origin[0],origin[1],origin[2]
        pC=torch.tensor([[x+x0,y+y0,z+z0] for x in np.arange(-((length/2)-0.5),length/2,1) for y in np.arange(-((length/2)-0.5),length/2,1) for z in np.arange(-((length/2)-0.5),length/2,1)])
        for i in range(len(pC)-1,-1,-1):
            if len(atoms_tree.query_radius(pC[i].numpy().reshape(1,3), r=0.5*math.sqrt(3))[0])!=0:
                pC = torch.cat((pC[:i],pC[i+1:]))
        return pC
    def calcVDW(self,protein_atoms,atoms_tree,pC):  
        # first col is C,then N,O,H
        vdws=np.zeros(shape=(len(pC), 2))
        for i in range(len(pC)):
            vdw_c,vdw_n,vdw_o,vdw_h=0,0,0,0
            
            for ind in (atoms_tree.query_radius(pC[i].numpy().reshape(1,3), r=10)[0]):
                (sigma1,epsilon1)=ff18SB[atom2vdwatom[(protein_atoms[ind].get_parent().get_resname(),protein_atoms[ind].get_full_id()[4][0])]]
                r=self.calcEuclidean(torch.tensor(protein_atoms[ind].get_coord())-pC[i])
                (sigma2,epsilon2)=(1.9080,0.1094)
                vdw_c=vdw_c+self.vdwEnergy(sigma1,1.9080,epsilon1,0.1094,r)
                #vdw_n=vdw_n+self.vdwEnergy(sigma1,1.8240,epsilon1,0.1700,r)
                #vdw_o=vdw_o+self.vdwEnergy(sigma1,1.6612,epsilon1,0.2100,r)
                vdw_h=vdw_h+self.vdwEnergy(sigma1,1.4870,epsilon1,0.0157,r)

            #vdws[i]=[vdw_c,vdw_n,vdw_o,vdw_h]
            vdws[i]=[vdw_c,vdw_h]
        return torch.from_numpy(vdws)
    def vdwEnergy(self,sigma1,sigma2,epsilon1,epsilon2,r):
        d = (sigma1+sigma2)/r
        d2 = d*d
        d4 = d2*d2
        d6 = d4*d2
        d12 = d6*d6
        e = math.sqrt(epsilon1*epsilon2)
        return 4.0*e*(0.25*d12-0.50*d6)    
    def calcEuclidean (self,p):
        return math.sqrt(p[0]**2+p[1]**2+p[2]**2)
    
    def normalize(self, scale = 1.0):
        """ Normalizes the point cloud to [-scale, scale].
          Args:
            scale (float): The scale factor.default to 1.0
        """
        min_v=-1*scale
        max_v=scale
        return torch.tensor(np.interp(self.psedoAtom,(self.psedoAtom.min(), self.psedoAtom.max()), (min_v, max_v)))
    
    
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
        self.psedoAtom = self.psedoAtom.to(self.device) @ rot

