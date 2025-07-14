from VoxelProt.surface2Octree.octree_generation_helper import *
slice_index=(1150,1200)
octree_address = "/../masif/"
data_address = "/../masif/"
octreeGeneration(data_address, octree_address,slice_index,threshold = 4,numberSelectCandidates = 11, data_type = "masif_data",binding_site = True,device="cuda")#data_type = "masif_data" or "chen11"

"""
#working on chen11
slice_index=(50,100)
octree_address = "/home/jingbo/chen11/"
data_address = "/media/jingbo/HD1_8TB/voxelprot_cofactor/chen11/"
numberSelectCandidates = 7
octreeGeneration(data_address, octree_address,slice_index,threshold = 4,numberSelectCandidates = numberSelectCandidates, data_type = "chen11",binding_site = False,device="cuda")
"""
