from VoxelProt.trainModel.train_model_helper import *
model_type = "vgg"  #or'ResNet' or  "SE_ResNet"
epoches=5
n_fold = 2
bindingoctree_add = "/home/jingbo/masif/BS_octree_cg/"
nonbindingoctree_add = "/home/jingbo/masif/nonBS_octree_cg/"
candidate_number= 11
lr=0.001
device="cuda"
checkpoint_path = None
train_model(model_type,epoches,n_fold,bindingoctree_add,nonbindingoctree_add,lr,candidate_number,checkpoint_path,device)

"""
if train chen11
from VoxelProt.trainModel.train_model_helper import *
model_type = "ResNet"  #or'ResNet' or  "SE_ResNet"
epoches=20
chen11_octree_add = "/home/jingbo/chen11/"
joined_octree_add = "/home/jingbo/joined/"
lr=0.001
device="cuda"
checkpoint_path = None
train_model_chen11(model_type,epoches,chen11_octree_add,joined_octree_add,lr,checkpoint_path,device)
"""
