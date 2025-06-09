from VoxelProt.dataset.dataset_helper import *
download_dir = "pdbdownload"
#1. get the experiment protein id
pdb_ids = get_protein_id()
#2. download pdb_ids under download_dir
download_pdbs(pdb_ids,download_dir)
#3. reduce software
reduce(download_dir)
#4.extract selected chains
output_pdb_contain_selected_chain("/home/jingbo/pdbdownload/filtered_chains_with_ligand/","/home/jingbo/pdbdownload/withH/")
#5. split it into protein and ligand
split_prot_ligand("/home/jingbo/masif/filtered_chains_with_ligand")
#6.run pdb2pqr
run_pdb2pqr("/home/jingbo/masif/split_proteins","/home/jingbo/masif/pqrs" )
#7. run abps
run_apbs("/home/jingbo/masif/pqrs" , "/home/jingbo/masif/dx_outputs")
#8. check_unknown_AA
check_unknown_AA("/home/jingbo/masif/split_proteins/")
#9. cross valiation
create_cross_val_splits(5)
