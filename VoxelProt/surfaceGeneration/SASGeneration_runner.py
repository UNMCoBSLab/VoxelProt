from VoxelProt.surfaceGeneration.SASFeature2CSV import *
pdb_address = "/../masif/split_proteins"
feature_address = "/../masif/features/"
dx_address = "/../dx_outputs/"
slice_index = (0,50)
csv_type = "masif_data"

SESGeneration(pdb_address,feature_address,dx_address,slice_index,csv_type)

