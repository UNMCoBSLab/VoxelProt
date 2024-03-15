#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


def positiveOctreeNoEnergy(protein,pdbAddress,featureaddress,octreeadd,device="cuda"):
    """
    protein="1a27"
    pdbAddress="/home/llab/Downloads/1a27.pdb"
    featureaddress="/home/llab/Downloads/features/"
    octreeadd="/home/llab/Downloads/BS_octree_N/"
    device="cuda"
    """
    addSurface=featureaddress+"surfaceNormals/"+protein+".csv"
    addKD=featureaddress+"KH/"+protein+".csv"
    addElec=featureaddress+"electro_info/"+protein+".csv"
    addCandidates=featureaddress+"candidates/"+protein+".csv"
    addAtom=featureaddress+"atomtype/"+protein+".csv"
    
    # parser a protein
    struc_dict = PDBParser(QUIET=True).get_structure(protein, pdbAddress)
    #get all atoms info with non-standard
    atoms = Selection.unfold_entities(struc_dict, "A")   
    #split all atoms into two groups, protein_atoms and cofactor_atoms
    ligands=["ACO","ACP","ADP","AMP","ANP","ATP","CAA","CDP","CMP","COA",
             "FAD","FMN","GDP","GMP","GNP","GTP","H4B","SCA","TPP","UMP",
             "NAP","NDP","NAI","NAD"]    
    cofactor_atoms=[item for item in atoms if item.get_parent().get_resname() in ligands]
    protein_atoms=[item for item in atoms if item.get_parent().get_resname() in k]    
    
    #get the residue name of protein_atoms
    protein_atoms_res=[item.get_parent().get_resname() for item in protein_atoms]
    #get the corr info of protein_atoms
    protein_atoms_coor=torch.tensor(np.array([item.get_coord() for item in protein_atoms]))
    #get the corr info of cofactor_atoms
    cofactor_atoms_coor=torch.tensor(np.array([item.get_coord() for item in cofactor_atoms]))    
    #get the feature info
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates= readData(addSurface,addKD,addElec,addCandidates,addAtom)

    #----------------around the NAP binding sites
    ns = Bio.PDB.NeighborSearch(cofactor_atoms)
    bindingPoints=[]
    for ind in range(len(surfacePoints)):
        if len(ns.search(surfacePoints[ind], 3))>0:
            bindingPoints.append(surfacePoints[ind])
    bindingPoints=torch.tensor([xx.tolist() for xx in bindingPoints])
    bindingPoints=subsample(bindingPoints,scale=4.0)
        
    #----------------
    kdt = KDTree(surfacePoints.detach().numpy(), metric='euclidean')
    #---------------for each bindingpoints, find the index of the most closet surfacepoint
    #----------------using the local coordinate of this point to create the box
    for n in range(len(bindingPoints)):
            nearest_ind=kdt.query(torch.reshape(bindingPoints[n], (1, 3)).detach().numpy(),k=1,return_distance=False)[:,0][0]
            #---------------create octree around the NAP binding site surface
            NAP_clouds=Points(surfacePoints[nearest_ind],surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),torch.tensor([1]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=16)
            NAP=Octree(depth=4,device = device)
            NAP.build_octree(NAP_clouds)
            NAP.build_neigh()
            saveOctree(NAP,octreeadd+protein+str(n)+"0.pkl")
            for i in range(0,3):
                NAP_clouds.rotate(90,"x")
                NAP=Octree(depth=4,device = device)
                NAP.build_octree(NAP_clouds)
                NAP.build_neigh()
                saveOctree(NAP,octreeadd+protein+str(n)+str(i+1)+".pkl")
            NAP_clouds.rotate(90,"x")
            for i in range(0,3):
                NAP_clouds.rotate(90,"y")
                NAP=Octree(depth=4,device = device)
                NAP.build_octree(NAP_clouds)
                NAP.build_neigh()
                saveOctree(NAP,octreeadd+protein+str(n)+str(i+4)+".pkl")
            NAP_clouds.rotate(90,"y")
            for i in range(0,3):
                NAP_clouds.rotate(90,"z")
                NAP=Octree(depth=4,device = device)
                NAP.build_octree(NAP_clouds)
                NAP.build_neigh()
                saveOctree(NAP,octreeadd+protein+str(n)+str(i+7)+".pkl")

