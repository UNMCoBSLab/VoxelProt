from VoxelProt.surface2Octree.octree_generation_helper import *
slice_index=(1150,1200)
octree_address = "/../masif/"
data_address = "/../masif/"
octreeGeneration(data_address, octree_address,slice_index,threshold = 4,numberSelectCandidates = 11 ,binding_site = False, device="cuda") 
