#this contains  channels C H
import torch
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors,KDTree
from VoxelProt.surface2Octree.vdwEnergy import *
from scipy.spatial.distance import cdist
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
            surroundAtoms=atoms_tree.query_radius(pC[i].numpy().reshape(1,3), r=10)[0]
            if(len(surroundAtoms)==0):
                vdws[i]=[0.0,0.0]
                continue
           
            
            sigma1=np.zeros(shape=(len(surroundAtoms), 1))
            sigma2c=np.full((len(surroundAtoms), 1), 1.9080)
            sigma2h=np.full((len(surroundAtoms), 1), 1.4870)    
            epsilon1=np.zeros(shape=(len(surroundAtoms), 1))
            epsilon2c=np.full((len(surroundAtoms), 1), 0.1094)
            epsilon2h=np.full((len(surroundAtoms), 1), 0.0157)
            trueAtoms=np.zeros(shape=(len(surroundAtoms), 3))
            fakeAtoms=np.full((len(surroundAtoms), 3), pC[i])

            for indes in range(len(surroundAtoms)):
                ind=surroundAtoms[indes]
                siep=ff18SB[atom2vdwatom[(protein_atoms[ind].get_parent().get_resname(),protein_atoms[ind].get_full_id()[4][0])]]
                sigma1[indes]=siep[0]
                epsilon1[indes]=siep[1]
                trueAtoms[indes]=protein_atoms[ind].get_coord()

            sigma12c=sigma1+sigma2c
            sigma12h=sigma1+sigma2h

            ec=np.sqrt(epsilon1*epsilon2c)
            eh=np.sqrt(epsilon1*epsilon2h)

            dc = (sigma12c)/(cdist(trueAtoms, fakeAtoms,'euclidean')[:, 0:1])               

            d2c=(dc**2)
            d4c = d2c*d2c
            d6c = d4c*d2c
            d12c = d6c*d6c    
            vdw_c=(4.0*ec*(0.25*d12c-0.50*d6c) ).sum(axis=0)

            dh = (sigma12h)/(cdist(trueAtoms, fakeAtoms,'euclidean')[:, 0:1])               
            d2h=(dh**2)
            d4h = d2h*d2h
            d6h = d4h*d2h
            d12h = d6h*d6h    
            vdw_h=(4.0*eh*(0.25*d12h-0.50*d6h)).sum(axis=0)
            vdws[i]=[vdw_c[0],vdw_h[0]]
        
        return torch.from_numpy(vdws)
           
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

