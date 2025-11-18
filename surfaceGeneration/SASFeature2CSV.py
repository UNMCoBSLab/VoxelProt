import Bio
from Bio.PDB import *
from VoxelProt.surfaceGeneration.dictionary import *
import numpy as np
from VoxelProt.surfaceGeneration.surfaceSAS import *
from VoxelProt.surfaceGeneration.helpers import *
import torch
from VoxelProt.surfaceGeneration.atomtype import *
from VoxelProt.surfaceGeneration.electrostatics import *
from VoxelProt.surfaceGeneration.hydrogen_bond_potential import *
from VoxelProt.surfaceGeneration.hydropathy_info import *
from gridData import Grid
from VoxelProt.surfaceGeneration.curvatures import *
import csv,os
from tqdm import tqdm
from itertools import islice

def SESGeneration_chen11(pdb_address,feature_address,dx_address,slice_index):
    pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "chen11-prt2lig.csv")
    
    first_col = []
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: 
                continue
            first_col.append(f"prot_{row[0]}")
            
    slice_pdb_list = first_col[slice_index[0]:slice_index[1]]    
    for pdbId in tqdm(slice_pdb_list): 
        SESfea2CSV(pdbId,f"{pdb_address}/{pdbId}",feature_address,f"{dx_address}/{pdbId[:-4]}.pqr.dx")
        
def SESGeneration_jointed(pdb_address,feature_address,dx_address,slice_index):
    pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "jointed-prt2lig.csv")
    
    first_col = []
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: 
                continue
            first_col.append(f"{row[0]}")
       
    slice_pdb_list = first_col[slice_index[0]:slice_index[1]]    
    for pdbId in tqdm(slice_pdb_list): 
        SESfea2CSV(pdbId,f"{pdb_address}/{pdbId}",feature_address,f"{dx_address}/{pdbId[:-4]}.pqr.dx")
        
def SESGeneration_coach420(pdb_address,feature_address,dx_address,slice_index):
    pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_all-prt2lig.csv")
    
    first_col = []
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: 
                continue
            first_col.append(f"{row[0]}")
       
    slice_pdb_list = first_col[slice_index[0]:slice_index[1]]    
    for pdbId in tqdm(slice_pdb_list): 
        SESfea2CSV(pdbId,f"{pdb_address}/{pdbId}",feature_address,f"{dx_address}/{pdbId[:-4]}.pqr.dx")
        
def SESGeneration_HOLO4K(pdb_address,feature_address,dx_address,slice_index):
    pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "HOLO4K-prot2lig.csv")
    
    first_col = []
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: 
                continue
            first_col.append(f"{row[0]}")
       
    slice_pdb_list = first_col[slice_index[0]:slice_index[1]]    
    for pdbId in tqdm(slice_pdb_list): 
        SESfea2CSV(pdbId,f"{pdb_address}/{pdbId}",feature_address,f"{dx_address}/{pdbId[:-4]}.pqr.dx")
        
        
def SESGeneration(pdb_address,feature_address,dx_address,slice_index,csv_type = "masif_data"):
    if csv_type == "masif_data":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "pdb_list_experiments.csv")
    elif csv_type == "coach":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_cofactor.csv")
    elif csv_type == "chen":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "chen_cofactor.csv")
    dict_data={}
    with open(pdb_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            dict_data[lines[0].split("_")[0]]=lines[0].split("_")[1]
    slice_dict_data = dict(islice(dict_data.items(), slice_index[0], slice_index[1]))
    for pdbId,chain in tqdm(slice_dict_data.items()): 
        SESfea2CSV(pdbId,f"{pdb_address}/prot_{pdbId}_{chain}.pdb",feature_address,f"{dx_address}/prot_{pdbId}_{chain}.pqr.dx")

    
def SESfea2CSV(pdbId,pdb_address,feature_address,dx_address):
    # parser a protein
    struc_dict = PDBParser(QUIET=True).get_structure(pdbId,pdb_address)
    #get all atoms info with non-standard
    atoms = Selection.unfold_entities(struc_dict, "A") 
    #get the info of protein_atoms        
    protein_atoms=[item for item in atoms if (item.get_parent().get_resname() in k)]
    protein_atoms_coor=np.array([item.get_coord() for item in protein_atoms])
    protein_atoms_res=[item.get_parent().get_resname() for item in protein_atoms]
    atomtypes_protein = cal_atomtype_radius(protein_atoms)

    # using the protein_atoms to create the surface model
    surfacePoints,candidate_index,shape_index,oriented_nor_vector,shape_index3 = createSurface (protein_atoms_coor,atomtypes_protein,sup_sampling=5,scales=[1.5],batch=None)

    #surfaceNormals,candidates,shape_index,shape_index3
    feathers2CSV(torch.cat((surfacePoints, oriented_nor_vector), 1).cpu(),feature_address+"surfaceNormals/",pdbId)
    feathers2CSV(candidate_index.cpu(),feature_address+"candidates/",pdbId)
    feathers2CSV(shape_index.cpu(),feature_address+"shape_index/",pdbId)  
    feathers2CSV(shape_index3.cpu(),feature_address+"shape_index3/",pdbId)

    g=Grid(dx_address)
    elec=getPoiBol(g,surfacePoints.cpu()) 
    feathers2CSV(elec,feature_address+"electro_info/",pdbId)
        
    protein_atoms_coor=torch.tensor(protein_atoms_coor)
    #atomtypes
    atomtype_surface=getAtomType(protein_atoms, protein_atoms_res,protein_atoms_coor,surfacePoints.cpu())
    feathers2CSV(atomtype_surface,feature_address+"atomtype/",pdbId)

    #KH
    KD=getKD(protein_atoms_coor,protein_atoms_res,surfacePoints.cpu())
    Hpotential=computeCharge(surfacePoints.cpu(), protein_atoms_coor, protein_atoms)
    KD_Hpotential=torch.cat((KD, Hpotential), 1)
    feathers2CSV(KD_Hpotential,feature_address+"KH/",pdbId)


    
    


