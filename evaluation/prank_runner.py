import os
import numpy as np
from Bio.PDB import *
from VoxelProt.evaluation.eval_helpers import *
import pandas as pd
def eval_per_protein_p2rank(protein_id, true_binding_site_pdb_path,p2rank_output, input_pdb_file, extra_top = 0):    
    number_true_sites = get_binding_site_number(protein_id[0])
    all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"{protein_id[0]}_{n}.pdb") for n in range(number_true_sites)]

    #get the predict
    #read predicted csv
    pred_fn = os.path.join(p2rank_output, f"prot_{protein_id[0]}_{protein_id[1]}.pdb_predictions.csv")
    if not os.path.isfile(pred_fn):
        return np.array([0]), np.array([20])
            
    pred_df = pd.read_csv(pred_fn, skipinitialspace=True)
        
    pred_df.columns = pred_df.columns.str.strip()
    #read the input pdb
    pdb_file = os.path.join(input_pdb_file, f"prot_{protein_id[0]}_{protein_id[1]}.pdb")
    structure = PDBParser(QUIET=True).get_structure("protein", pdb_file)
    all_atoms = [atom for atom in structure.get_atoms()]        
    # extract atoms for each pocket
    pockets = dict()
    for i, row in pred_df.iterrows():
        if i <= number_true_sites+extra_top:
            pocket_id = int(row["name"].strip().replace('pocket', ''))
            atom_ids = list(map(int, str(row["surf_atom_ids"]).split()))
            atoms = [all_atoms[idx] for idx in atom_ids if idx < len(all_atoms)]
            pockets[pocket_id] = atoms
    dvo_lst, dcc_lst = [], []
    
    for pdb_true in all_pdb_true:
        dvo_temp,dcc_temp = [],[]
        
        for _,predicted_binding_site in pockets.items():
            try:
                dvo = compute_dvo(pdb_true, predicted_binding_site)
                dcc = compute_dcc(pdb_true, predicted_binding_site) 
                dvo_temp.append(dvo)
                dcc_temp.append(dcc)
            except:
                pass
        if len(dcc_temp) !=0:
            min_dcc = min(dcc_temp)    
            min_index = dcc_temp.index(min_dcc)  
            dvo_lst.append(dvo_temp[min_index])
            dcc_lst.append(min_dcc) 
        else:
            dcc_lst .append(40.0)         
            dvo_lst .append(0.0)
            
    return np.array(dvo_lst) ,np.array(dcc_lst)
def p2rank_eval(n_fold, p2rank_output, true_binding_site_pdb_path, input_pdb_file, extra_top=0,csv_type = "masif_data"): 
    dvo_rv = np.array([])  
    dcc_rv = np.array([])
    test_list = get_test_id(n_fold,csv_type)
        
    for protein_id in test_list:
        dvo_np, dcc_np = eval_per_protein_p2rank(protein_id, true_binding_site_pdb_path,p2rank_output, input_pdb_file, extra_top)
            
        dvo_rv = np.concatenate([dvo_rv, dvo_np])
        dcc_rv = np.concatenate([dcc_rv, dcc_np])

    return dvo_rv, dcc_rv
    


input_pdb_file = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/split_proteins/"
p2rank_output = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/results/p2rank/p2rank_output"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/true_binding_site"
csv_type = "masif_data"
for n_fold in range(1,6):
    for extra_top in [0,2]:
        dvo_rv,dcc_rv = p2rank_eval(n_fold,p2rank_output,true_binding_site_pdb_path,input_pdb_file,extra_top,csv_type = csv_type)
        plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),p2rank)",save_path = f"{csv_type}_p2rank (n + {extra_top})_{n_fold}") 
        mean,std = stat_dvo(dcc_rv, dvo_rv, cutoff = 4)
        print(f"fold{n_fold},extra_top{extra_top}: {mean} +/- {std}")
        np.savetxt(f'dvo_fold{n_fold}_extraTop{extra_top}.csv', dvo_rv, delimiter=',')
        np.savetxt(f'dcc_fold{n_fold}_extraTop{extra_top}.csv', dcc_rv, delimiter=',')
        
"""

input_pdb_file = "/media/jingbo/HD1_8TB/voxelprot_cofactor/chen_cofactor/split_proteins/"
p2rank_output = "/media/jingbo/HD1_8TB/voxelprot_cofactor/chen_cofactor/results/p2rank/p2rank_output"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/chen_cofactor/true_binding_site/"
csv_type = "chen_cofactor"  #masif_data or coach_cofactor or chen_cofactor
n_fold = 0

for extra_top in [0,2]:
    dvo_rv,dcc_rv = p2rank_eval(n_fold,p2rank_output,true_binding_site_pdb_path,input_pdb_file,extra_top,csv_type = csv_type)
    plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),p2rank)",save_path = f"{csv_type}_p2rank (n + {extra_top})_{n_fold}") 
    mean,std = stat_dvo(dcc_rv, dvo_rv, cutoff = 4)
    print(f"fold{n_fold},extra_top{extra_top}: {mean} +/- {std}")
    np.savetxt(f'dvo_fold{n_fold}_extraTop{extra_top}.csv', dvo_rv, delimiter=',')
    np.savetxt(f'dcc_fold{n_fold}_extraTop{extra_top}.csv', dcc_rv, delimiter=',')

"""
"""
import os
import numpy as np
from Bio.PDB import *
from VoxelProt.evaluation.eval_helpers import *
import pandas as pd
from tqdm import tqdm
def eval_per_protein_p2rank(protein_id, true_ligand_pdb_path, true_binding_site_pdb_path,p2rank_output, input_pdb_file, csv_type="masif_data",extra_top = 0):    
    if csv_type=="coach420_all"  or csv_type=="coach420_excluded" or csv_type=="HOLO4K_excluded" or csv_type=="HOLO4K_all": 
        number_true_sites = get_binding_site_number(protein_id,csv_type)
        ligand_pdb_true = [os.path.join(true_ligand_pdb_path, f"lig_{protein_id}_{n}.pdb") for n in range(number_true_sites)]
        all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"prot_{protein_id}.pdb_{n}.pdb") for n in range(number_true_sites)]
        pred_fn = os.path.join(p2rank_output, f"prot_{protein_id}.pdb_predictions.csv")

        if not os.path.isfile(pred_fn): return np.array([0]), np.array([20])
                
        pred_df = pd.read_csv(pred_fn, skipinitialspace=True)
            
        pred_df.columns = pred_df.columns.str.strip()
        #read the input pdb
        pdb_file = os.path.join(input_pdb_file, f"prot_{protein_id}.pdb")
        
    else:
        number_true_sites = get_binding_site_number(protein_id[0],csv_type)
        all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"{protein_id[0]}_{n}.pdb") for n in range(number_true_sites)]
        #get the predict
        #read predicted csv
        pred_fn = os.path.join(p2rank_output, f"prot_{protein_id[0]}_{protein_id[1]}.pdb_predictions.csv")
        if not os.path.isfile(pred_fn):
            return np.array([0]), np.array([20])
                
        pred_df = pd.read_csv(pred_fn, skipinitialspace=True)
            
        pred_df.columns = pred_df.columns.str.strip()
        #read the input pdb
        pdb_file = os.path.join(input_pdb_file, f"prot_{protein_id[0]}_{protein_id[1]}.pdb")

        
    structure = PDBParser(QUIET=True).get_structure("protein", pdb_file)
    all_atoms = [atom for atom in structure.get_atoms()]        
    # extract atoms for each pocket
    pockets = dict()
    for i, row in pred_df.iterrows():
        if i <= number_true_sites+extra_top:
            pocket_id = int(row["name"].strip().replace('pocket', ''))
            atom_ids = list(map(int, str(row["surf_atom_ids"]).split()))
            atoms = [all_atoms[idx] for idx in atom_ids if idx < len(all_atoms)]
            pockets[pocket_id] = atoms

    dvo_lst, dcc_lst,dca_lst = [], [], []
    
    for ind in range(number_true_sites):
        pdb_true = all_pdb_true[ind]
        pdb_ligand = ligand_pdb_true[ind]        
        dvo_temp,dcc_temp,dca_temp = [],[],[]
                           
        for _,pdb_pred in pockets.items():
            dvo = compute_dvo(pdb_true, pdb_pred)
            dcc = compute_dcc(pdb_true, pdb_pred)     
            dca = compute_dca(pdb_ligand, pdb_pred)                 
            dvo_temp.append(dvo)
            dcc_temp.append(dcc)
            dca_temp.append(dca)
                
        if len(dca_temp) !=0:
            min_dcc = min(dcc_temp)
            min_dca = min(dca_temp)
            min_index = dca_temp.index(min_dca)  
            dvo_lst.append(dvo_temp[min_index])
            dcc_lst.append(min_dcc) 
            dca_lst.append(min_dca)
        else:
            dcc_lst .append(40.0)         
            dvo_lst .append(0.0)
            dca_lst.append(40.0)        
    return np.array(dvo_lst) ,np.array(dcc_lst),np.array(dca_lst)

    
def p2rank_eval(n_fold, p2rank_output, true_ligand_pdb_path,true_binding_site_pdb_path, input_pdb_file, extra_top=0,csv_type = "masif_data"): 
    dvo_rv = np.array([])  
    dcc_rv = np.array([])
    dca_rv = np.array([])
    test_list = get_test_id(n_fold,csv_type)
        
    for protein_id in tqdm(test_list):
        #print(protein_id)
        dvo_np, dcc_np, dca_np = eval_per_protein_p2rank(protein_id, true_ligand_pdb_path, true_binding_site_pdb_path,p2rank_output, input_pdb_file, csv_type, extra_top)
        dvo_rv = np.concatenate([dvo_rv, dvo_np])
        dcc_rv = np.concatenate([dcc_rv, dcc_np])
        dca_rv = np.concatenate([dca_rv, dca_np])

    return dvo_rv, dcc_rv, dca_rv

input_pdb_file = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_proteins/"
p2rank_output = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/results/p2rank/p2rank_output"
true_ligand_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_ligands_single(all)/"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/true_binding_site(all)/"
csv_type = "HOLO4K_all"  #masif_data or coach_cofactor or chen_cofactor or coach420_all or coach420_excluded, HOLO4K_all
n_fold = 0
for extra_top in [0,2]:
    dvo_rv, dcc_rv, dca_rv = p2rank_eval(n_fold,p2rank_output,true_ligand_pdb_path,true_binding_site_pdb_path,input_pdb_file,extra_top,csv_type = csv_type)

    plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),p2rank)",save_path = f"DCC_{csv_type}_p2rank (n + {extra_top})_{n_fold}") 
    plot_dcc_curve(dca_rv, f"{csv_type}_DCA (fold{n_fold},top (n + {extra_top}),p2rank)",save_path = f"DCA_{csv_type}_p2rank (n + {extra_top})_{n_fold}") 

    
    mean,std = stat_dvo(dca_rv, dvo_rv, cutoff = 4)    
    print(f"fold{n_fold},extra_top{extra_top}: {mean} +/- {std}")
    
    np.savetxt(f'dvo_fold{n_fold}_extraTop{extra_top}.csv', dvo_rv, delimiter=',')
    np.savetxt(f'dcc_fold{n_fold}_extraTop{extra_top}.csv', dcc_rv, delimiter=',')
    np.savetxt(f'dca_fold{n_fold}_extraTop{extra_top}.csv', dca_rv, delimiter=',')"""
