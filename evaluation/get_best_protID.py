import csv,os
from VoxelProt.evaluation.eval_helpers import *
def read_csv(file):
    rows = []
    with open(file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)      
        for row in reader:
            rows.append(float(row[0]))
    return rows
    
def read_csvs(result_dir):    
    dca_file = os.path.join(result_dir,f"dca_fold0_extraTop0.csv")
    dcc_file = os.path.join(result_dir,f"dcc_fold0_extraTop0.csv")
    dvo_file = os.path.join(result_dir,f"dvo_fold0_extraTop0.csv")
    
    dca_rows=read_csv(dca_file)
    dcc_rows=read_csv(dcc_file)
    dvo_rows=read_csv(dvo_file)
    return dca_rows,dcc_rows,dvo_rows
def get_best_protID(csv_type,result_dir):
    dca,dcc,dvo = read_csvs(result_dir)
    
    test_list = get_test_id(0,csv_type)
    ind = -1
    for protein_id in test_list:
        is_good = False
        mark = 0
        number_true_sites = get_binding_site_number(protein_id,csv_type)
        for i in range(number_true_sites):
            ind=ind+1        
            if dca[ind]<=4 and dcc[ind]<=4 and dvo[ind]>=0.5:
                is_good=True
                mark=mark+1
        if is_good:
            print(protein_id,f"{mark}/{number_true_sites}")  
            
            
csv_type =  "coach420_all"  # "coach420_all" or  "coach420_excluded" or csv_type=="HOLO4K_all" or csv_type=="HOLO4K_excluded"
result_dir = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/coachresults/voxelprot/all/"
get_best_protID(csv_type,result_dir)            
