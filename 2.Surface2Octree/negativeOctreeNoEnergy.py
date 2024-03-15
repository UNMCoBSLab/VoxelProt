#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Bio
from Bio.PDB import *
import numpy as np
import torch
import pykeops
from pykeops.torch import LazyTensor
from pickleoctree import *
from dictionary import *
from readdata import *
from sklearn.neighbors import KDTree
from points import Points
from octree import Octree
import random


# In[2]:


def subsample(x, batch=None, scale=1.0):
    if batch is None:  # Single protein case:
            labels = pykeops.torch.cluster.grid_cluster(x, scale).long()
            C = labels.max() + 1
            # We append a "1" to the input vectors, in order to
            # compute both the numerator and denominator of the "average"
            #  fraction in one pass through the data.
            x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
            D = x_1.shape[1]
            points = torch.zeros_like(x_1[:C])
 
            points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
           
    return (points[:, :-1] / points[:, -1:]).contiguous()


# In[ ]:


def negativeOctreeNoEnergy(protein,pdbAddress,featureaddress,octree_add,numberSelectCandidates=15,device="cuda"):
    """
    #this is used to generate the nonbinding site Octrees
    protein="1a27"
    pdbAddress="/home/llab/Downloads/1a27.pdb"
    featureaddress="/home/llab/Downloads/features/"
    octree_add= "/home/llab/Downloads/nonBS_octree_N/"
    device="cuda"
    numberSelectCandidates=15   
    """
    addSurface=featureaddress+"surfaceNormals/"+protein+".csv"
    addKD=featureaddress+"KH/"+protein+".csv"
    addElec=featureaddress+"electro_info/"+protein+".csv"
    addCandidates=featureaddress+"candidates/"+protein+".csv"
    addAtom=featureaddress+"atomtype/"+protein+".csv"

    #get the feature info 
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates= readData(addSurface,addKD,addElec,addCandidates,addAtom) 
    # parser a protein
    struc_dict = PDBParser(QUIET=True).get_structure(protein, pdbAddress)
    #get all atoms info with non-standard
    atoms = Selection.unfold_entities(struc_dict, "A")

    #split all atoms into two groups, protein_atoms and cofactor_atoms
    protein_atoms=[item for item in atoms if item.get_parent().get_resname() in k]
    #split all atoms into two groups, protein_atoms and cofactor_atoms
    ligands=["ACO","ACP","ADP","AMP","ANP","ATP","CAA","CDP","CMP","COA",
             "FAD","FMN","GDP","GMP","GNP","GTP","H4B","SCA","TPP","UMP",
             "NAP","NDP","NAI","NAD"]
    for each in ligands:
        vars()["cofactor_atoms_"+each] =[item for item in atoms if item.get_parent().get_resname() ==each]
        if len(vars()["cofactor_atoms_"+each])!=0:
            vars()["ns_"+each] =  Bio.PDB.NeighborSearch( (vars()["cofactor_atoms_"+each]) ) 

    #----------------nonNAP binding site    
    #find all negative candidates 
    candidates=subsample(torch.stack(candidates),scale=4.0)  
    no_centers_points=[]
    for ind in range(len(candidates)):
        add=True
        for each in ligands:    
            try:
                if len((vars()["ns_"+each]).search(candidates[ind], 4))!=0:
                    add=False
                    break
            except:
                pass
        if add:
            no_centers_points.append(candidates[ind]) 
    #---only store some number of candidates
    if len(no_centers_points)>numberSelectCandidates:
        no_centers_points=random.sample(no_centers_points,numberSelectCandidates)  
    else:
        numberSelectCandidates=len(no_centers_points)  

    #find another 9 octrees within each box
    N=[[]for each in range(numberSelectCandidates)]    

    dd=8 
    # 
    for p in surfacePoints: 
        ifadd=True
        for each in ligands:
            try:     
                if len( (vars()["ns_"+each]).search(p, 4) )!=0:
                    ifadd=False
                    break
            except:
                pass
        if ifadd:           
            x,y,z=p[0].item(),p[1].item(),p[2].item()                                
            for index in range(len(no_centers_points)):                
                if (abs(x-no_centers_points[index][0].item())<=dd) and (abs(y-no_centers_points[index][1].item())<=dd) and (abs(z-no_centers_points[index][2].item())<=dd):
                    N[index].append(p)         

    kdt = KDTree(surfacePoints.detach().numpy(), metric='euclidean')                        
    for index in range(len(N)):
        if len(N[index])>=9:
            N[index]=random.sample(N[index],9)  
        nearest_ind=kdt.query(torch.reshape(no_centers_points[index], (1, 3)).detach().numpy(),k=1,return_distance=False)[:,0][0]
        N[index].append(surfacePoints[nearest_ind]) 

    #-------------------------------------------  
    # create octree around the non-NAP binding site center      
    for n in range(len(N)):
        for ind in range(len(N[n])):
            NonNAP_clouds=Points(N[n][ind],surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),torch.tensor([0]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=16)
            NonNAP=Octree(depth=4,device = device)
            NonNAP.build_octree(NonNAP_clouds)
            NonNAP.build_neigh()
            saveOctree(NonNAP,octree_add+protein+str(n)+str(ind)+"0.pkl")
            for i in range(0,3):
                NonNAP_clouds.rotate(90,"x")
                NonNAP=Octree(depth=4,device = device)
                NonNAP.build_octree(NonNAP_clouds)
                NonNAP.build_neigh()
                saveOctree(NonNAP,octree_add+protein+str(n)+str(ind)+str(i+1)+".pkl")
            NonNAP_clouds.rotate(90,"x")
            for i in range(0,3):
                NonNAP_clouds.rotate(90,"y")
                NonNAP=Octree(depth=4,device = device)
                NonNAP.build_octree(NonNAP_clouds)
                NonNAP.build_neigh()
                saveOctree(NonNAP,octree_add+protein+str(n)+str(ind)+str(i+4)+".pkl")
            NonNAP_clouds.rotate(90,"y")
            for i in range(0,3):
                NonNAP_clouds.rotate(90,"z")
                NonNAP=Octree(depth=4,device = device)
                NonNAP.build_octree(NonNAP_clouds)
                NonNAP.build_neigh()
                saveOctree(NonNAP,octree_add+protein+str(n)+str(ind)+str(i+7)+".pkl")

