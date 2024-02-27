#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import random

def surface2negativeOctree(protein,pdbAddress,chainid,numberSelectCandidates,addSurface,addKD,addElec,addCandidates,addAtom,ligands,chem_geo_octree_add,vdw_octree_add,device):
    """
    protein="3bwc" 
    pdbAddress="/home/llab/Desktop/JBLab/detection/MasifDataset/ent/pdb"+protein+".ent"
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

    chem_geo_octree_add="/home/llab/Desktop/JBLab/detection/MasifDataset/nonBS_octree_cg1/"
    vdw_octree_add="/home/llab/Desktop/JBLab/detection/MasifDataset/nonBS_octree_v1/"
    device="cuda"

    numberSelectCandidates=15
    """    
    #get the feature info 
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates= readData(addSurface,addKD,addElec,addCandidates,addAtom) 

    # parser a protein
    struc_dict = PDBParser(QUIET=True).get_structure(protein,pdbAddress)
    #get all atoms info with non-standard
    atoms = Selection.unfold_entities(struc_dict, "A")  

    #split all atoms into two groups, protein_atoms and cofactor_atoms
    protein_atoms=[item for item in atoms if (item.get_full_id()[2] in chainid) and (item.get_parent().get_resname() in k)]
    atoms_tree = KDTree(np.array([a.get_coord() for a in protein_atoms]), leaf_size=2) 
    
    for each in ligands:
        vars()["cofactor_atoms_"+each] =[a for a in atoms if (a.get_full_id()[2] in chainid) and (a.get_parent().get_resname() ==each)] 
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
            chem_geo_clouds=Points(N[n][ind],surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),torch.tensor([0]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=16)
            chem_geo_octree=Octree(depth=4,device = device)
            chem_geo_octree.build_octree(chem_geo_clouds)
            chem_geo_octree.build_neigh()
            saveOctree(chem_geo_octree,chem_geo_octree_add+protein+str(n)+str(ind)+"0.pkl")          

            vdw_clouds=VDWPoints(protein_atoms,atoms_tree,N[n][ind],device,length=16)
            vdw_octree=VDWOctree(depth=4,device = device)
            vdw_octree.build_octree(vdw_clouds)
            vdw_octree.build_neigh()
            saveOctree(vdw_octree,vdw_octree_add+protein+str(n)+str(ind)+"0.pkl")                

            for i in range(0,3):
                chem_geo_clouds.rotate(90,"x")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()
                saveOctree(chem_geo_octree,chem_geo_octree_add+protein+str(n)+str(ind)+str(i+1)+".pkl")

                vdw_clouds.rotate(90,"x")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()
                saveOctree(vdw_octree, vdw_octree_add+protein+str(n)+str(ind)+str(i+1)+".pkl")


            chem_geo_clouds.rotate(90,"x")
            vdw_clouds.rotate(90,"x")                                                                                                                                 

            for i in range(0,3):
                chem_geo_clouds.rotate(90,"y")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()
                saveOctree(chem_geo_octree,chem_geo_octree_add+protein+str(n)+str(ind)+str(i+4)+".pkl")

                vdw_clouds.rotate(90,"y")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()
                saveOctree(vdw_octree,vdw_octree_add+protein+str(n)+str(ind)+str(i+4)+".pkl")

            chem_geo_clouds.rotate(90,"y")
            vdw_clouds.rotate(90,"y")             

            for i in range(0,3):
                chem_geo_clouds.rotate(90,"z")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()
                saveOctree(chem_geo_octree,chem_geo_octree_add+protein+str(n)+str(ind)+str(i+7)+".pkl")       


                vdw_clouds.rotate(90,"z")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()
                saveOctree(vdw_octree,vdw_octree_add+protein+str(n)+str(ind)+str(i+7)+".pkl")                         


# In[ ]:




