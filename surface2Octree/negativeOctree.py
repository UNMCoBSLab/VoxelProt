import Bio
from Bio.PDB import *
import numpy as np
import os,random
from sklearn.neighbors import KDTree
from VoxelProt.surface2Octree.dictionary import *
from VoxelProt.surface2Octree.readdata import *
from VoxelProt.surface2Octree.pickleoctree import *
from VoxelProt.surface2Octree.helpers import *
from VoxelProt.surface2Octree.VDW2Points import VDWPoints
from VoxelProt.surface2Octree.VDW2Octree import VDWOctree
from VoxelProt.surface2Octree.points_SAS import Points
from VoxelProt.surface2Octree.octree import Octree

def surface2negativeOctree(pdbId,chain,threshold,pdbAddress,ligAddress,numberSelectCandidates,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,octreeadd_v,device):
    # get protein atoms    
    struc_dict_p = PDBParser(QUIET=True).get_structure("protein",pdbAddress)
    atoms_p = Selection.unfold_entities(struc_dict_p, "A") 
    protein_atoms=[item for item in atoms_p if item.get_parent().get_resname() in k]
    
    # get ligand atoms        
    struc_dict_l = PDBParser(QUIET=True).get_structure("ligand",ligAddress)
    atoms_l = Selection.unfold_entities(struc_dict_l, "A")   
    cofactor_atoms=[item for item in atoms_l if item.get_parent().get_resname() in COFACTOR]
    all_cofactor = list(set([item.get_parent().get_resname() for item in cofactor_atoms]))
    
    #get the feature info 
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates= readData(addSurface,addKD,addElec,addCandidates,addAtom)      

    #----------------
    #get the corr info of protein_atoms and create a atoms_tree
    protein_atoms_coor=torch.tensor(np.array([item.get_coord() for item in protein_atoms]))
    atoms_tree = KDTree(protein_atoms_coor.numpy(), leaf_size=2) 
    
    
    for each in all_cofactor:
        vars()["cofactor_atoms_"+each] =[a for a in cofactor_atoms if a.get_parent().get_resname() ==each] 
        vars()["ns_"+each] =  Bio.PDB.NeighborSearch( (vars()["cofactor_atoms_"+each]) ) 
            

    #----------------nonNAP binding site    
    #find all negative candidates 
    candidates=subsample(torch.stack(candidates),scale=4.0)  
    no_centers_points=[]
    for ind in range(len(candidates)):
        add=True
        for each in all_cofactor: 
            if len((vars()["ns_"+each]).search(candidates[ind], threshold*2))!=0:
                add=False
                break
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
        for each in all_cofactor:
            if len( (vars()["ns_"+each]).search(p, 4) )!=0:
                ifadd=False
                break
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
            saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(ind)+"0.pkl")          

            vdw_clouds=VDWPoints(protein_atoms,atoms_tree,N[n][ind],device,length=16)
            vdw_octree=VDWOctree(depth=4,device = device)
            vdw_octree.build_octree(vdw_clouds)
            vdw_octree.build_neigh()
            saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+str(ind)+"0.pkl")                

            for i in range(0,3):
                chem_geo_clouds.rotate(90,"x")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()
                saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(ind)+str(i+1)+".pkl")

                vdw_clouds.rotate(90,"x")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()
                saveOctree(vdw_octree, octreeadd_v+pdbId+str(n)+str(ind)+str(i+1)+".pkl")


            chem_geo_clouds.rotate(90,"x")
            vdw_clouds.rotate(90,"x")                                                                                                                                 

            for i in range(0,3):
                chem_geo_clouds.rotate(90,"y")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()
                saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(ind)+str(i+4)+".pkl")

                vdw_clouds.rotate(90,"y")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()
                saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+str(ind)+str(i+4)+".pkl")

            chem_geo_clouds.rotate(90,"y")
            vdw_clouds.rotate(90,"y")             

            for i in range(0,3):
                chem_geo_clouds.rotate(90,"z")
                chem_geo_octree=Octree(depth=4,device = device)
                chem_geo_octree.build_octree(chem_geo_clouds)
                chem_geo_octree.build_neigh()
                saveOctree(chem_geo_octree,octreeadd_cg+pdbId+str(n)+str(ind)+str(i+7)+".pkl")       


                vdw_clouds.rotate(90,"z")
                vdw_octree=VDWOctree(depth=4,device = device)
                vdw_octree.build_octree(vdw_clouds)
                vdw_octree.build_neigh()
                saveOctree(vdw_octree,octreeadd_v+pdbId+str(n)+str(ind)+str(i+7)+".pkl")                         

