import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from pathlib import Path
import os,csv
def read_dcc_list(csv_path):
    data = [] 
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(float(row[0]))
    
    return data  
def read_HOLO_list(csv_type):
    if csv_type=="HOLO4K_all":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "HOLO4K_all-prot2lig.csv")
    elif csv_type=="HOLO4K_excluded":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "HOLO4K_excluded-prot2lig.csv")  
        
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        rv = []
        for protein, lig_str in reader:
            rv.append(protein[5:9])
    return rv
def read_coach420_list(csv_type):
    if csv_type=="coach420_all":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_all-prt2lig.csv")
    elif csv_type=="coach420_excluded":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_excluded-prt2lig.csv")  
        
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        rv = []
        for protein, lig_str in reader:
            rv.append(protein[5:10])
    return rv
    
def get_test_id(n_fold,csv_type = "masif_data"):
    if csv_type=="coach420_all" or csv_type=="coach420_excluded":
        return read_coach420_list(csv_type)
    if csv_type=="HOLO4K_all" or csv_type=="HOLO4K_excluded":
        return read_HOLO_list(csv_type)
        
    ids = []
    if csv_type == "masif_data":
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "cross_val_splits",f"fold_{n_fold}","test.txt")
    elif csv_type == "coach_cofactor":
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_cofactor.csv")
    elif csv_type == "chen_cofactor":
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "chen_cofactor.csv")
    with open(file_dir, newline='') as f:
        reader = csv.reader(f)          
        for row in reader:
            full = row[0]               
            code, chain = full.split('_', 1) 
            ids.append((code,chain))
    
    return ids 
    
 
def get_binding_site_number(target_protein_id,csv_type="masif_data"):
    if csv_type=="coach420_all" :
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "bindingSiteNumber_coach420all.csv")
    elif csv_type=="coach420_excluded":     
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "bindingSiteNumber_coach420excluded.csv") 
    elif csv_type=="HOLO4K_all":     
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "bindingSiteNumber_HOLO4Kall.csv") 
    elif csv_type=="HOLO4K_excluded":     
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "bindingSiteNumber_HOLO4Kexcluded.csv")         
    else:
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "bindingSiteNumber.csv")
        
    with open(file_dir, newline='') as f:
        reader = csv.reader(f)     
        for row in reader:
            if row[0].lower() == target_protein_id.lower():
                return int(row[1] )
        return -1  
        
def load_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    atoms = struct.get_atoms()
    coords = np.array([atom.get_coord() for atom in atoms])
    return coords

def coords_to_voxels(coords, origin,voxel_size = 1.0):
    """
    Map each (x,y,z) coordinate to a voxel index triple (i,j,k).
    """
    shifted = (coords - origin) / voxel_size
    indices = np.floor(shifted).astype(int)
    return {tuple(idx) for idx in indices}

def compute_dvo(pdb_true, pdb_pred, voxel_size=1.0):
    # load coordinates
    coords_true = load_coords(pdb_true)
    if isinstance(pdb_pred, list):
        coords_pred = np.array([atom.get_coord() for atom in pdb_pred])
    else:
        coords_pred = load_coords(pdb_pred)

    # choose grid origin
    all_coords  = np.vstack([coords_true, coords_pred])
    origin = all_coords.min(axis=0)  

    # compute intersection & union
    vox_true = coords_to_voxels(coords_true, origin, voxel_size)
    vox_pred = coords_to_voxels(coords_pred, origin, voxel_size)
    inter = vox_true & vox_pred
    uni = vox_true | vox_pred

    dvo = len(inter) / len(uni) if uni else 0.0
    return dvo   
def compute_dcc(pdb_true, pdb_pred):
    # load coordinates
    coords_true = load_coords(pdb_true)
    if isinstance(pdb_pred, list):
        coords_pred = np.array([atom.get_coord() for atom in pdb_pred])
    else:
        coords_pred = load_coords(pdb_pred)

    pred_center = coords_pred.mean(axis=0)   # (3,)
    true_center = coords_true.mean(axis=0)   # (3,)

    dcc = np.linalg.norm(pred_center - true_center)
    return dcc

def compute_euc_dist(detected_center,center_pre):
    return np.linalg.norm(detected_center - center_pre)
    
def stat_dvo(dcc_rv, dvo_rv, cutoff = 4):
    mask = dcc_rv < cutoff
    filtered_dvo = dvo_rv[mask]
    return np.mean(filtered_dvo),np.std(filtered_dvo)


def plot_multiple_curves(dcc_lists, labels, title, save_path=None,  thresholds=np.linspace(0, 10, 100)):
    plt.figure(figsize=(6, 4))
    for dcc_list, lbl in zip(dcc_lists, labels):
        dcc_array = np.array(dcc_list)
        # drop NaNs
        dcc_array = dcc_array[~np.isnan(dcc_array)]
        # compute cumulative success rate at each threshold
        cumulative = [np.mean(dcc_array <= t) for t in thresholds]
        # plot
        plt.plot(thresholds, cumulative, marker='o', lw=2, label=lbl)

    plt.xlabel('DCC Threshold (Å)')
    plt.ylabel('Cumulative Fraction')
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(thresholds[0], thresholds[-1])
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    
def plot_dcc_curve(dcc_list, title, save_path = None, thresholds=np.linspace(0, 10, 100)):
    dcc_array = np.array(dcc_list)
    dcc_array = dcc_array[~np.isnan(dcc_array)]  

    cumulative = [np.mean(dcc_array <= t) for t in thresholds]

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, cumulative, marker='o', lw=2)
    plt.xlabel('DCC Threshold (Å)')
    plt.ylabel('Cumulative Fraction')
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(thresholds[0], thresholds[-1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    plt.show()    

def compute_dca(pdb_ligand, pdb_pred):
    #get the pred_pocket_center
    if isinstance(pdb_pred, list):
        coords_pred = np.array([atom.get_coord() for atom in pdb_pred])
    else:
        coords_pred = load_coords(pdb_pred)

    pred_center = coords_pred.mean(axis=0)   # (3,)
    
    #get ligands
    structure = PDBParser(QUIET=True).get_structure("ligand", pdb_ligand)

    lig_coords = []
    for atom in structure.get_atoms():
        elem = (atom.element or "").upper().strip()
        if elem == "H": continue
        lig_coords.append(atom.coord)

    lig_coords = np.vstack(lig_coords)
    # Euclidean distances to the center, take minimum
    dists = np.linalg.norm(lig_coords - pred_center, axis=1)
    return float(dists.min())
    
def eval_per_true_binding_site(pdb_true,pdb_pred,pdb_ligand):  
    predicted_binding_site = [atom for atom in PDBParser(QUIET=True).get_structure("p",pdb_pred).get_atoms()]
    dvo = compute_dvo(pdb_true, predicted_binding_site)
    dcc = compute_dcc(pdb_true, predicted_binding_site)  
    dca = compute_dca(pdb_ligand, predicted_binding_site)  
    return dvo,dcc,dca 
    

