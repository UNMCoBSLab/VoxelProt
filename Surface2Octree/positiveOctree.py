#!/usr/bin/env python
# coding: utf-8

# In[8]:


import Bio
from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree

from dictionary import *
from readdata import *
from pickleoctree import *
from helpers import *
from VDW2Points import VDWPoints
from VDW2Octree import VDWOctree
from points_SAS import Points
from octree import Octree


# In[ ]:


def surface2postiveOctree(protein,pdbAddress,chainid,addSurface,addKD,addElec,addCandidates,addAtom,ligands,octreeadd_cg,octreeadd_v,device):
    """
    protein="5jr7"
    pdbAddress="/home/llab/Desktop/JBLab/detection/MasifDataset/ent/pdb5jr7.ent"
    chainid="AB"

    addSurface="/home/llab/Desktop/JBLab/detection/MasifDataset/features/surfaceNormals/"+protein+".csv"
    addKD="/home/llab/Desktop/JBLab/detection/MasifDataset/features/KH/"+protein+".csv"
    addElec="/home/llab/Desktop/JBLab/detection/MasifDataset/features/electro_info/"+protein+".csv"  
    addCandidates="/home/llab/Desktop/JBLab/detection/MasifDataset/features/candidates/"+protein+".csv"   
    addAtom="/home/llab/Desktop/JBLab/detection/MasifDataset/features/atomtype/"+protein+".csv"  

    ligands=["HEM","HEC","HAS","HEO","D97","DHE","SAM","SAH","ACO","COD",
    "ACP","ADP","AMP","ANP","ATP","CDP","COA","FAD","FMN","CDM",
    "GDP","GMP","GTP","H4B","NAD","NAP","NDP","TPP","UMP","5GP",
    "COO","CTP","DGT","EPU","FUH","GAP","GKD","GKE","GTG","BRU",
    "HMG","HP7","HXC","JB2","MCN","MD1","MDE","MGD","MJZ","APR",
    "N01","OXT","PAP","PPS","PRX","ST9","T3Q","TH3","TLO","AGS",
    "TXP","TYD","UAG","UD1","UDP","UDX","UMA","UPG","ADQ","ADX","A1S",]
    
    octreeadd_cg="/home/llab/Desktop/JBLab/detection/MasifDataset/BS_octree_cg/"
    octreeadd_v="/home/llab/Desktop/JBLab/detection/MasifDataset/BS_octree_v/"

    device="cuda"
    """
    #get the feature info 
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates= readData(addSurface,addKD,addElec,addCandidates,addAtom)      

    # parser a protein
    struc_dict = PDBParser(QUIET=True).get_structure(protein,pdbAddress)
    #get all atoms info with non-standard
    atoms = Selection.unfold_entities(struc_dict, "A") 

    #split all atoms into two groups, protein_atoms and cofactor_atoms
    protein_atoms=[item for item in atoms if (item.get_full_id()[2] in chainid) and (item.get_parent().get_resname() in k)]
    cofactor_atoms=[item for item in atoms if item.get_parent().get_resname() in ligands]

    #get the corr info of protein_atoms and create a atoms_tree
    protein_atoms_coor=torch.tensor(np.array([item.get_coord() for item in protein_atoms]))
    atoms_tree = KDTree(protein_atoms_coor.numpy(), leaf_size=2) 

    #----------------find all binding surface points around the binding sites
    ns = Bio.PDB.NeighborSearch(cofactor_atoms)
    bindingPoints=[item for item in surfacePoints if len(ns.search(item, 3))>0 ]
    bindingPoints=torch.tensor([xx.tolist() for xx in bindingPoints])
    bindingPoints=subsample(bindingPoints,scale=4.0) 

    #----------------
    kdt = KDTree(surfacePoints.detach().numpy(), metric='euclidean')
    #---------------for each bindingpoints, find the index of the most closet surfacepoint
    #----------------using the local coordinate of this point to create the box
    for n in range(len(bindingPoints)):
            nearest_ind=kdt.query(torch.reshape(bindingPoints[n], (1, 3)).detach().numpy(),k=1,return_distance=False)[:,0][0]
            vdw_clouds=VDWPoints(protein_atoms,atoms_tree,surfacePoints[nearest_ind],device,length=16)
            vdw_octree=VDWOctree(depth=4,device = device)
            vdw_octree.build_octree(vdw_clouds)
            vdw_octree.build_neigh()

            chem_geo_clouds=Points(surfacePoints[nearest_ind],surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),torch.tensor([1]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=16)
            chem_geo_octree=Octree(depth=4,device = device)
            chem_geo_octree.build_octree(chem_geo_clouds)  
            chem_geo_octree.build_neigh()

            saveOctree(chem_geo_octree,octreeadd_cg+protein+str(n)+"0.pkl")
            saveOctree(vdw_octree,octreeadd_v+protein+str(n)+"0.pkl")

            for i in range(0,3):                    
                vdw_clouds.rotate(90,"x")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()

                chem_geo_clouds.rotate(90,"x")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()

                saveOctree(chem_geo_octree,octreeadd_cg+protein+str(n)+str(i+1)+".pkl")
                saveOctree(vdw_octree,octreeadd_v+protein+str(n)+str(i+1)+".pkl")

            chem_geo_clouds.rotate(90,"x")
            vdw_clouds.rotate(90,"x")

            for i in range(0,3):
                vdw_clouds.rotate(90,"y")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()

                chem_geo_clouds.rotate(90,"y")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()

                saveOctree(chem_geo_octree,octreeadd_cg+protein+str(n)+str(i+4)+".pkl")
                saveOctree(vdw_octree,octreeadd_v+protein+str(n)+str(i+4)+".pkl")

            chem_geo_clouds.rotate(90,"y")
            vdw_clouds.rotate(90,"y")

            for i in range(0,3):
                vdw_clouds.rotate(90,"z")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()

                chem_geo_clouds.rotate(90,"z")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()


                saveOctree(chem_geo_octree,octreeadd_cg+protein+str(n)+str(i+7)+".pkl")       
                saveOctree(vdw_octree,octreeadd_v+protein+str(n)+str(i+7)+".pkl")   

