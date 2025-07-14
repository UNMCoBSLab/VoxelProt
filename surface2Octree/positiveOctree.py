import Bio
from Bio.PDB import *
import numpy as np
import os
from sklearn.neighbors import KDTree
from VoxelProt.surface2Octree.dictionary import *
from VoxelProt.surface2Octree.readdata import *
from VoxelProt.surface2Octree.pickleoctree import *
from VoxelProt.surface2Octree.helpers import *
#from VoxelProt.surface2Octree.VDW2Points import VDWPoints
#from VoxelProt.surface2Octree.VDW2Octree import VDWOctree
from VoxelProt.surface2Octree.points_SAS import Points
from VoxelProt.surface2Octree.octree import Octree


def surface2postiveOctree(pdbId,threshold,pdbAddress,ligAddress,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,octreeadd_v=None,device="cuda",data_type = "masif_data"):
    # get protein atoms    
    struc_dict_p = PDBParser(QUIET=True).get_structure("protein",pdbAddress)
    atoms_p = Selection.unfold_entities(struc_dict_p, "A") 
    if data_type == "masif_data":
        protein_atoms=[item for item in atoms_p if item.get_parent().get_resname() in k]
    elif data_type == "chen11" or data_type == "joined":
        protein_atoms=[item for item in atoms_p]
        
    # get ligand atoms 
    if data_type == "chen11" or data_type == "joined":
        atoms_l = []
        cofactor_atoms = []
        for liga in ligAddress:
            try:
                struc_dict_l = PDBParser(QUIET=True).get_structure("ligand",liga)
                atoms_l += Selection.unfold_entities(struc_dict_l, "A")   
                cofactor_atoms += [item for item in atoms_l]
            except:
                pass
    
    elif data_type == "masif_data": 
        struc_dict_l = PDBParser(QUIET=True).get_structure("ligand",ligAddress)
        atoms_l = Selection.unfold_entities(struc_dict_l, "A")   
        cofactor_atoms=[item for item in atoms_l if item.get_parent().get_resname() in COFACTOR]
    
    #get the feature info 
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates= readData(addSurface,addKD,addElec,addCandidates,addAtom)      

    #----------------find all binding surface points around the binding sites
    ns = Bio.PDB.NeighborSearch(cofactor_atoms)
    bindingPoints=[item for item in surfacePoints if len(ns.search(item, threshold))>0 ]
    bindingPoints=torch.tensor([xx.tolist() for xx in bindingPoints])
    bindingPoints=subsample(bindingPoints,scale=4.0) 

    #----------------
    #get the corr info of protein_atoms and create a atoms_tree
    protein_atoms_coor=torch.tensor(np.array([item.get_coord() for item in protein_atoms]))
    atoms_tree = KDTree(protein_atoms_coor.numpy(), leaf_size=2) 
    kdt = KDTree(surfacePoints.detach().numpy(), metric='euclidean')
    #---------------for each bindingpoints, find the index of the most closet surfacepoint
    #----------------using the local coordinate of this point to create the box
            
    for n in range(len(bindingPoints)):
            nearest_ind=kdt.query(torch.reshape(bindingPoints[n], (1, 3)).detach().numpy(),k=1,return_distance=False)[:,0][0]
            
            #vdw_clouds=VDWPoints(protein_atoms,atoms_tree,surfacePoints[nearest_ind],device,length=16)
            #vdw_octree=VDWOctree(depth=4,device = device)
            #vdw_octree.build_octree(vdw_clouds)
            #vdw_octree.build_neigh()

            chem_geo_clouds=Points(surfacePoints[nearest_ind],surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),torch.tensor([1]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=16)
            chem_geo_octree=Octree(depth=4,device = device)
            chem_geo_octree.build_octree(chem_geo_clouds)  
            chem_geo_octree.build_neigh()

            saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+"0.pkl")
            #saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+"0.pkl")

            for i in range(0,3):                    
                #vdw_clouds.rotate(90,"x")
                #vdw_octree=VDWOctree(depth=4,device = device)
                #vdw_octree.build_octree(vdw_clouds)
                #vdw_octree.build_neigh()

                chem_geo_clouds.rotate(90,"x")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()

                saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(i+1)+".pkl")
                #saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+str(i+1)+".pkl")

            chem_geo_clouds.rotate(90,"x")
            #vdw_clouds.rotate(90,"x")

            for i in range(0,3):
                #vdw_clouds.rotate(90,"y")
                #vdw_octree=VDWOctree(depth=4,device = device)
                #vdw_octree.build_octree(vdw_clouds)
                #vdw_octree.build_neigh()

                chem_geo_clouds.rotate(90,"y")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()

                saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(i+4)+".pkl")
                #saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+str(i+4)+".pkl")

            chem_geo_clouds.rotate(90,"y")
            #vdw_clouds.rotate(90,"y")

            for i in range(0,3):
                #vdw_clouds.rotate(90,"z")
                #vdw_octree=VDWOctree(depth=4,device = device)
                #vdw_octree.build_octree(vdw_clouds)
                #vdw_octree.build_neigh()

                chem_geo_clouds.rotate(90,"z")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()


                saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(i+7)+".pkl")       
                #saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+str(i+7)+".pkl")   

