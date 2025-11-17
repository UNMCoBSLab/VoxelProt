from VoxelProt.evaluation.get_true_binding_site_helper import *

#step1 process all ligands

input_dirt = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_ligands/"
distance_cutoff = 4.0
exc = {'FE','MN','CA','SO4','LU','K','PO4','PO3','AF3','O','F','NH4','TL','CO3','CL','PT','XE','MG',
'OH','AL','NA','CO','CS','ZN','SF4','H2S','CAD','YB','HG','NO','SB','CU','SO2','BO3','2HP',
'VXA','GD3','6MO','FSX','DOD','FES','BF2','CO2','MOS','BR','CAD','CMO','FEO','0QE','HGI',
'PEO','FE2','VO4','POP','NO3','MOH','NCO','NI','TE','F3S','HDN','TAS','ARS','OMO','AZI',
'4MO','OXY','MMC','SBO','CD','IUM','FNE','IOD','CAC','DPO','SO3','SR','SM','YT3','PI','BEF',
'HOA','MO','MGF','CYN','UNX','RB','GD','ALF','NH2','2MO','3PO','VA3'}

process_ligand(input_dirt, distance_cutoff = distance_cutoff,exc=exc,print_out=False)

#step 2 get all binding sites
csv_type="HOLO4K_all"
threshold = 4
addSurface = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/features/surfaceNormals/"
lid_dir = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_ligands_single(all)/"
protein_dir = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/split_proteins/"
out_fn_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/HOLO4K/true_binding_site(all)/"

get_true_binding_site(protein_dir,lid_dir,out_fn_path,addSurface,csv_type = csv_type,threshold = threshold)
