from VoxelProt.dataset.dataset_helper import *
download_dir = "pdbdownload"
csv_type = "chen_cofactor" # or others

#1. get the experiment protein id
pdb_ids = get_protein_id(csv_file = csv_type) #"masif_data","coach_cofactor","chen_cofactor"
#2. download pdb_ids under download_dir
download_pdbs(pdb_ids,download_dir)

#3. reduce software
reduce(download_dir)

#4.extract selected chains
output_pdb_contain_selected_chain("/home/jingbo/pdbdownload/filtered_chains_with_ligand/","/home/jingbo/pdbdownload/withH/",csv_type)

#5. split it into protein and ligand
split_prot_ligand("/home/jingbo/chen/filtered_chains_with_ligand")

#6.run pdb2pqr
run_pdb2pqr("/home/jingbo/chen/split_proteins","/home/jingbo/chen/pqrs" )
#7. run abps
run_apbs("/home/jingbo/chen/pqrs" , "/home/jingbo/chen/dx_outputs")

#8. check_unknown_AA
check_unknown_AA("/home/jingbo/chen/split_proteins/")
#9. cross valiation
create_cross_val_splits(5)

