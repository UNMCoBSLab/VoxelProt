import os
import numpy as np
from Bio.PDB import *
from VoxelProt.evaluation.eval_helpers import *
def eval_per_protein_fpocket(protein_id, true_binding_site_pdb_path,voxelprot_output, extra_top = 0):    
    number_true_sites = get_binding_site_number(protein_id[0])
    all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"{protein_id[0]}_{n}.pdb") for n in range(number_true_sites)]
    pred_pdb_fn = [os.path.join(fpocket_output, f"prot_{protein_id[0]}_{protein_id[1]}_out", "pockets", f"pocket{n+1}_atm.pdb") for n in range(number_true_sites+extra_top)]
    dvo_lst, dcc_lst = [], []
    for pdb_true in all_pdb_true:
        dvo_temp,dcc_temp = [],[]
        for pdb_pred in pred_pdb_fn:
            try:
                dvo,dcc = eval_per_true_binding_site(pdb_true,pdb_pred)
                dvo_temp.append(dvo)
                dcc_temp.append(dcc)
            except:
                pass
        min_dcc = min(dcc_temp)    
        min_index = dcc_temp.index(min_dcc)  
        dvo_lst.append(dvo_temp[min_index])
        dcc_lst.append(min_dcc) 
    return np.array(dvo_lst) ,np.array(dcc_lst)
    
def fpocket_eval(n_fold, fpocket_output, true_binding_site_pdb_path, extra_top=0,csv_type = "masif_data"): 
    dvo_rv = np.array([])  
    dcc_rv = np.array([])
    test_list = get_test_id(n_fold,csv_type)
        
    for protein_id in test_list:
        dvo_np, dcc_np = eval_per_protein_fpocket(protein_id, true_binding_site_pdb_path, fpocket_output, extra_top)
        dvo_rv = np.concatenate([dvo_rv, dvo_np])
        dcc_rv = np.concatenate([dcc_rv, dcc_np])

    return dvo_rv, dcc_rv
    
"""    
fpocket_output = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/results/fpocket/fpockets_output"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/true_binding_site"
csv_type = "masif_data"
for n_fold in range(1,6):
    for extra_top in [0,2]:
        dvo_rv,dcc_rv = fpocket_eval(n_fold,fpocket_output,true_binding_site_pdb_path,extra_top,csv_type)
        plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),fpocket)",save_path = f"{csv_type}_fpocket_top (n + {extra_top})_{n_fold}") 
        mean,std = stat_dvo(dcc_rv, dvo_rv, cutoff = 4)
        print(f"fold{n_fold},extra_top{extra_top}: {mean} +/- {std}")
        np.savetxt(f'dvo_fold{n_fold}_extraTop{extra_top}.csv', dvo_rv, delimiter=',')
        np.savetxt(f'dcc_fold{n_fold}_extraTop{extra_top}.csv', dcc_rv, delimiter=',')
"""        
"""
fpocket_output = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach_cofactor/results/fpocket/fpockets_output"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach_cofactor/true_binding_site/"
csv_type = "coach_cofactor"  #masif_data or coach_cofactor or chen_cofactor
n_fold = 0
for extra_top in [0,2]:
    dvo_rv,dcc_rv = fpocket_eval(n_fold,fpocket_output,true_binding_site_pdb_path,extra_top,csv_type)
    plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),fpocket)",save_path = f"{csv_type}_fpocket_top (n + {extra_top})_{n_fold}") 
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
from tqdm import tqdm
def eval_per_protein_fpocket(protein_id, true_ligand_pdb_path,true_binding_site_pdb_path, csv_type="masif_data",extra_top = 0):   
    if csv_type=="coach420_all"  or csv_type=="coach420_excluded" or csv_type=="HOLO4K_excluded" or csv_type=="HOLO4K_all": 
        number_true_sites = get_binding_site_number(protein_id,csv_type)
        ligand_pdb_true = [os.path.join(true_ligand_pdb_path, f"lig_{protein_id}_{n}.pdb") for n in range(number_true_sites)]
        all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"prot_{protein_id}.pdb_{n}.pdb") for n in range(number_true_sites)]
        pred_pdb_fn = [os.path.join(fpocket_output, f"prot_{protein_id}_out", "pockets", f"pocket{n+1}_atm.pdb") for n in range(number_true_sites+extra_top)]
    else:
        number_true_sites = get_binding_site_number(protein_id[0],csv_type)
        all_pdb_true = [os.path.join(f"{true_binding_site_pdb_path}", f"{protein_id[0]}_{n}.pdb") for n in range(number_true_sites)]
        pred_pdb_fn = [os.path.join(fpocket_output, f"prot_{protein_id[0]}_{protein_id[1]}_out", "pockets", f"pocket{n+1}_atm.pdb") for n in range(number_true_sites+extra_top)]

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
        
        min_dcc = min(dcc_temp)
        min_dca = min(dca_temp)
        min_index = dca_temp.index(min_dca)  
        dvo_lst.append(dvo_temp[min_index])
        dcc_lst.append(min_dcc) 
        dca_lst.append(min_dca) 
        
    return np.array(dvo_lst) ,np.array(dcc_lst),np.array(dca_lst)
    
def fpocket_eval(n_fold, true_ligand_pdb_path,true_binding_site_pdb_path, extra_top=0, csv_type = "masif_data"): 
    dvo_rv = np.array([])  
    dcc_rv = np.array([])
    dca_rv = np.array([])
    
    test_list = get_test_id(n_fold,csv_type)
        
    for protein_id in tqdm(test_list):
        dvo_np, dcc_np, dca_np = eval_per_protein_fpocket(protein_id, true_ligand_pdb_path, true_binding_site_pdb_path, csv_type, extra_top)
        dvo_rv = np.concatenate([dvo_rv, dvo_np])
        dcc_rv = np.concatenate([dcc_rv, dcc_np])
        dca_rv = np.concatenate([dca_rv, dca_np])

    return dvo_rv, dcc_rv, dca_rv
    
fpocket_output = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/results/fpocket/fpockets_output"
true_ligand_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_ligands_single(all)/"
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/true_binding_site(all)/"
csv_type = "HOLO4K_all"  #masif_data or coach_cofactor or chen_cofactor or coach420_all or coach420_excluded,HOLO4K_all
n_fold = 0

for extra_top in [0,2]:
    dvo_rv, dcc_rv, dca_rv = fpocket_eval(n_fold,true_ligand_pdb_path, true_binding_site_pdb_path,extra_top,csv_type)
    plot_dcc_curve(dcc_rv, f"{csv_type}_DCC (fold{n_fold},top (n + {extra_top}),fpocket)",save_path = f"DCC_{csv_type}_fpocket_top (n + {extra_top})_{n_fold}") 
    plot_dcc_curve(dca_rv, f"{csv_type}_DCA (fold{n_fold},top (n + {extra_top}),fpocket)",save_path = f"DCA_{csv_type}_fpocket_top (n + {extra_top})_{n_fold}") 

    
    mean,std = stat_dvo(dca_rv, dvo_rv, cutoff = 4)
    
    print(f"fold{n_fold},extra_top{extra_top}: {mean} +/- {std}")
    
    np.savetxt(f'dvo_fold{n_fold}_extraTop{extra_top}.csv', dvo_rv, delimiter=',')
    np.savetxt(f'dcc_fold{n_fold}_extraTop{extra_top}.csv', dcc_rv, delimiter=',')
    np.savetxt(f'dca_fold{n_fold}_extraTop{extra_top}.csv', dca_rv, delimiter=',')
    
"""
