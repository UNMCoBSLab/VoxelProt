import os,csv
from VoxelProt.surface2Octree.negativeOctree import *
from VoxelProt.surface2Octree.positiveOctree import *
from tqdm import tqdm
from itertools import islice

def octreeGeneration(data_address, octree_address,slice_index,threshold,numberSelectCandidates = 11 ,binding_site = True,device="cuda"):
    #create the storage folder
    if binding_site:
        octreeadd_cg=f"{octree_address}BS_octree_cg/"
        octreeadd_v=f"{octree_address}BS_octree_v/"
    else:
        octreeadd_cg=f"{octree_address}nonBS_octree_cg/"
        octreeadd_v=f"{octree_address}nonBS_octree_v/"        
    os.makedirs(octreeadd_cg, exist_ok=True)
    os.makedirs(octreeadd_v, exist_ok=True)
    
    #the the pdb list and slice part of it 
    pdb_list =  os.path.join(os.getcwd(), "VoxelProt", "dataset", "pdb_list_experiments.csv")
    dict_data={}
    with open(pdb_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            dict_data[lines[0].split("_")[0]]=lines[0].split("_")[1]    
    slice_dict_data = dict(islice(dict_data.items(), slice_index[0], slice_index[1]))

    # create the octree
    for pdbId,chain in tqdm(slice_dict_data.items()):     
        pdbAddress=f"{data_address}split_proteins/prot_{pdbId}_{chain}.pdb" 
        ligAddress=f"{data_address}split_ligands/lig_{pdbId}_{chain}.pdb" 
        addSurface=f"{data_address}features/surfaceNormals/{pdbId}.csv"
        addKD=f"{data_address}features/KH/{pdbId}.csv"
        addElec=f"{data_address}features/electro_info/{pdbId}.csv"
        addCandidates=f"{data_address}features/candidates/{pdbId}.csv"  
        addAtom=f"{data_address}features/atomtype/{pdbId}.csv"  
        if binding_site:    
            surface2postiveOctree(pdbId,chain,threshold,pdbAddress,ligAddress,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,octreeadd_v,device)
        else:    
            surface2negativeOctree(pdbId,chain,threshold,pdbAddress,ligAddress,numberSelectCandidates,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,octreeadd_v,device)   
