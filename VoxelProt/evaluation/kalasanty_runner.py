import os
import numpy as np
from Bio.PDB import *
from VoxelProt.evaluation.eval_helpers import *
from tqdm import tqdm
def mol2_to_pdb_single(mol2_path):
    pdb_path = mol2_path.replace(".mol2", ".pdb")
    atoms = []
    with open(mol2_path) as f:
        in_atoms = False
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>") and not line.startswith("@<TRIPOS>ATOM"):
                in_atoms = False
            if in_atoms and line.strip():
                parts = line.split()
                atom_id = int(parts[0])
                atom_name = parts[1]
                x, y, z = map(float, parts[2:5])
                resname = parts[7][:3] if len(parts) > 7 else "UNK"
                resid = 1
                atoms.append((atom_id, atom_name, resname, "A", resid, x, y, z))

    # Write to PDB
    with open(pdb_path, "w") as out:
        for atom in atoms:
            out.write(f"ATOM  {atom[0]:5d} {atom[1]:<4} {atom[2]:>3} {atom[3]:1}{atom[4]:4d}    {atom[5]:8.3f}{atom[6]:8.3f}{atom[7]:8.3f}  1.00  0.00\n")
        out.write("END\n")
    return pdb_path
    
def mol2pdb(pred_path):
    for folder in os.listdir(pred_path):
        for i in range(50):
            try:
                mol2_path = os.path.join(pred_path,folder,'mol2_files',f"pocket{i}.mol2")    
                mol2_to_pdb_single(mol2_path)
            except:
                pass


def eval_per_protein_kalasanty(protein_id, fpocket_output,true_ligand_pdb_path,true_binding_site_pdb_path, csv_type="masif_data",extra_top = 0):   
    if csv_type=="coach420_all"  or csv_type=="coach420_excluded" or csv_type=="HOLO4K_excluded" or csv_type=="HOLO4K_all": 
        number_true_sites = get_binding_site_number(protein_id,csv_type)
        ligand_pdb_true = [os.path.join(true_ligand_pdb_path, f"lig_{protein_id}_{n}.pdb") for n in range(number_true_sites)]
        all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"prot_{protein_id}.pdb_{n}.pdb") for n in range(number_true_sites)]
        pred_pdb_fn = [os.path.join(fpocket_output, f"prot_{protein_id}", "mol2_files", f"pocket{n}.pdb") for n in range(number_true_sites+extra_top)]
        
    else:
        number_true_sites = get_binding_site_number(protein_id[0],csv_type)
        all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"{protein_id[0]}_{n}.pdb") for n in range(number_true_sites)]
        pred_pdb_fn = [os.path.join(fpocket_output, f"prot_{protein_id[0]}_{protein_id[1]}", "mol2_files", f"pocket{n}.pdb") for n in range(number_true_sites+extra_top)]
        
    dvo_lst, dcc_lst,dca_lst = [], [], []
    
    for ind in range(number_true_sites):
        pdb_true = all_pdb_true[ind]
        pdb_ligand = ligand_pdb_true[ind]
        dvo_temp,dcc_temp,dca_temp = [],[],[]
        for pdb_pred in pred_pdb_fn:
            try:
                dvo,dcc,dca = eval_per_true_binding_site(pdb_true,pdb_pred,pdb_ligand)
                dvo_temp.append(dvo)
                dcc_temp.append(dcc)
                dca_temp.append(dca)
            except:
                pass
        if len(dvo_temp)==0:
            dvo_lst.append(0)
            dcc_lst.append(40) 
            dca_lst.append(40)  
        else:
            min_dcc = min(dcc_temp)
            min_dca = min(dca_temp)
            min_index = dca_temp.index(min_dca)  
            dvo_lst.append(dvo_temp[min_index])
            dcc_lst.append(min_dcc) 
            dca_lst.append(min_dca) 
        
    return np.array(dvo_lst) ,np.array(dcc_lst),np.array(dca_lst)
    
def kalasanty_eval(n_fold, fpocket_output, true_ligand_pdb_path,true_binding_site_pdb_path, extra_top=0, csv_type = "masif_data"): 
    dvo_rv = np.array([])  
    dcc_rv = np.array([])
    dca_rv = np.array([])
    
    test_list = get_test_id(n_fold,csv_type)
        
    for protein_id in tqdm(test_list):
        dvo_np, dcc_np, dca_np = eval_per_protein_kalasanty(protein_id, fpocket_output, true_ligand_pdb_path, true_binding_site_pdb_path, csv_type, extra_top)
        dvo_rv = np.concatenate([dvo_rv, dvo_np])
        dcc_rv = np.concatenate([dcc_rv, dcc_np])
        dca_rv = np.concatenate([dca_rv, dca_np])

    return dvo_rv, dcc_rv, dca_rv
    
    
    
kalasanty_output = '/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/results/kalasanty/kalasanty_pockets/'
true_ligand_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_ligands_single(excluded)/"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/true_binding_site(excluded)/"
csv_type = "HOLO4K_excluded"  #masif_data or coach_cofactor or chen_cofactor or coach420_all or coach420_excluded,HOLO4K_all,HOLO4K_excluded
n_fold = 0

#mol2pdb(kalasanty_output)

for extra_top in [0,2]:
    dvo_rv, dcc_rv, dca_rv = kalasanty_eval(n_fold,kalasanty_output, true_ligand_pdb_path, true_binding_site_pdb_path,extra_top,csv_type)
    plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),kalasanty)",save_path = f"DCC_{csv_type}_fpocket_top (n + {extra_top})_{n_fold}") 
    plot_dcc_curve(dca_rv, f"{csv_type}_DCA (fold{n_fold},top (n + {extra_top}),kalasanty)",save_path = f"DCA_{csv_type}_fpocket_top (n + {extra_top})_{n_fold}") 

    
    mean,std = stat_dvo(dca_rv, dvo_rv, cutoff = 4)
    
    print(f"fold{n_fold},extra_top{extra_top}: {mean} +/- {std}")
    
    np.savetxt(f'dvo_fold{n_fold}_extraTop{extra_top}.csv', dvo_rv, delimiter=',')
    np.savetxt(f'dcc_fold{n_fold}_extraTop{extra_top}.csv', dcc_rv, delimiter=',')
    np.savetxt(f'dca_fold{n_fold}_extraTop{extra_top}.csv', dca_rv, delimiter=',')  
