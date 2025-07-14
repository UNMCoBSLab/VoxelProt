import os,csv
from VoxelProt.surface2Octree.negativeOctree import *
from VoxelProt.surface2Octree.positiveOctree import *
from tqdm import tqdm
from itertools import islice
def read_joined():
    path =  os.path.join(os.getcwd(), "VoxelProt", "dataset", "joined-prt2lig.csv")
    d = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            key = row[0]
            combined = ";".join(row[1:])
            values = [v for v in combined.split(";") if v]
            d[key] = values
    return d
    
def read_masif_data():
    pdb_list =  os.path.join(os.getcwd(), "VoxelProt", "dataset", "pdb_list_experiments.csv")
    dict_data={}
    with open(pdb_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            dict_data[lines[0].split("_")[0]]=lines[0].split("_")[1]    
    return dict_data
    
def read_chen11():
    path =  os.path.join(os.getcwd(), "VoxelProt", "dataset", "chen11-prt2lig.csv")
    d = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            key = row[0]
            combined = ";".join(row[1:])
            values = [v for v in combined.split(";") if v]
            d[key] = values
    return d
    
def octreeGeneration(data_address, octree_address,slice_index,threshold,numberSelectCandidates = 11, data_type = "masif_data",binding_site = True,device="cuda"):
    #create the storage folder
    if binding_site:
        octreeadd_cg=f"{octree_address}BS_octree_cg/"
        #octreeadd_v=f"{octree_address}BS_octree_v/"
    else:
        octreeadd_cg=f"{octree_address}nonBS_octree_cg/"
        #octreeadd_v=f"{octree_address}nonBS_octree_v/"        
    os.makedirs(octreeadd_cg, exist_ok=True)
    #os.makedirs(octreeadd_v, exist_ok=True)
    
    #the the pdb list and slice part of it 
    if data_type ==  "masif_data":
    
        dict_data = read_masif_data()
        slice_dict_data = dict(islice(dict_data.items(), slice_index[0], slice_index[1]))
        # create the octree
        for pdbId,chain in tqdm(slice_dict_data.items()):     
            pdbAddress = os.path.join(data_address, "split_proteins", f"prot_{pdbId}_{chain}.pdb")
            ligAddress = os.path.join(data_address, "split_ligands", f"lig_{pdbId}_{chain}.pdb")
            addSurface = os.path.join(data_address, "features", "surfaceNormals",f"{pdbId}.csv")
            addKD = os.path.join(data_address, "features", "KH",f"{pdbId}.csv")
            addElec = os.path.join(data_address, "features", "electro_info",f"{pdbId}.csv")
            addCandidates = os.path.join(data_address, "features", "candidates",f"{pdbId}.csv")
            addAtom = os.path.join(data_address, "features", "atomtype",f"{pdbId}.csv")
            
            if binding_site:    
                surface2postiveOctree(pdbId,threshold,pdbAddress,ligAddress,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,device = device)
            else:    
                surface2negativeOctree(pdbId,threshold,pdbAddress,ligAddress,numberSelectCandidates,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,device = device)
                
                
    # for training model
    if data_type ==  "chen11":                
        dict_data = read_chen11()
        slice_dict_data = dict(islice(dict_data.items(), slice_index[0], slice_index[1]))

        # create the octree
        for pdbId,ligs in tqdm(slice_dict_data.items()):     
            pdbAddress = os.path.join(data_address, "chen11_prot", f"prot_{pdbId}")
            ligAddress = [os.path.join(data_address, "chen11_lig", f"lig_{pdbId[:-4]}_{lig_ind}.pdb") for lig_ind in range(len(ligs))]
            addSurface = os.path.join(data_address, "features", "surfaceNormals",f"prot_{pdbId}.csv")
            addKD = os.path.join(data_address, "features", "KH",f"prot_{pdbId}.csv")
            addElec = os.path.join(data_address, "features", "electro_info",f"prot_{pdbId}.csv")
            addCandidates = os.path.join(data_address, "features", "candidates",f"prot_{pdbId}.csv")
            addAtom = os.path.join(data_address, "features", "atomtype",f"prot_{pdbId}.csv")
            
            if binding_site:    
                surface2postiveOctree(pdbId,threshold,pdbAddress,ligAddress,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,device = device,data_type = data_type)
            else:    
                surface2negativeOctree(pdbId,threshold,pdbAddress,ligAddress,numberSelectCandidates,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,device = device,data_type = data_type)
                
    # for training model
    if data_type ==  "joined":                
        dict_data = read_joined()
        slice_dict_data = dict(islice(dict_data.items(), slice_index[0], slice_index[1]))

        # create the octree
        for pdbId,ligs in tqdm(slice_dict_data.items()):     
            pdbAddress = os.path.join(data_address, "split_proteins", pdbId)
            ligAddress = [os.path.join(data_address, "split_ligands", f"lig_{pdbId[5:-4]}_{lig_ind}.pdb") for lig_ind in range(len(ligs))]
            addSurface = os.path.join(data_address, "features", "surfaceNormals",f"{pdbId}.csv")
            addKD = os.path.join(data_address, "features", "KH",f"{pdbId}.csv")
            addElec = os.path.join(data_address, "features", "electro_info",f"{pdbId}.csv")
            addCandidates = os.path.join(data_address, "features", "candidates",f"{pdbId}.csv")
            addAtom = os.path.join(data_address, "features", "atomtype",f"{pdbId}.csv")
            if binding_site:    
                surface2postiveOctree(pdbId,threshold,pdbAddress,ligAddress,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,device = device,data_type = data_type)
            else:    
                surface2negativeOctree(pdbId,threshold,pdbAddress,ligAddress,numberSelectCandidates,addSurface,addKD,addElec,addCandidates,addAtom,octreeadd_cg,device = device,data_type = data_type)
