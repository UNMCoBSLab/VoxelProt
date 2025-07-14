from VoxelProt.surfaceGeneration.SASFeature2CSV import *
pdb_address = "/../masif/split_proteins"
feature_address = "/../masif/features/"
dx_address = "/../dx_outputs/"
slice_index = (0,50)
csv_type = "masif_data"

SESGeneration(pdb_address,feature_address,dx_address,slice_index,csv_type)

"""
if run the chen11 dataset
from VoxelProt.surfaceGeneration.SASFeature2CSV import *
pdb_address = "/home/jingbo/VoxelProt/dataset/chen11/chen11_prot"
feature_address = "/home/jingbo/chen11/features/"
dx_address = "/home/jingbo/chen11/dx_outputs/"
slice_index = (50,100)

SESGeneration_chen11(pdb_address,feature_address,dx_address,slice_index)

if run the jointed dataset
from VoxelProt.surfaceGeneration.SASFeature2CSV import *
pdb_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/joined/split_proteins/"
feature_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/joined/features/"
dx_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/joined/dx_outputs/"
slice_index = (0,2)

SESGeneration_jointed(pdb_address,feature_address,dx_address,slice_index)

if run the coach420 dataset
from VoxelProt.surfaceGeneration.SASFeature2CSV import *
pdb_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/split_proteins/"
feature_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/features/"
dx_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/dx_outputs/"
slice_index = (0,50)

SESGeneration_coach420(pdb_address,feature_address,dx_address,slice_index)
"""

