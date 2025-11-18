from VoxelProt.trainModel.train_model_helper import *
model_type = "vgg"  #or'ResNet' or  "SE_ResNet"
epoches=5
n_fold = 0
bindingoctree_add = "/.../masif/BS_octree_cg/"
nonbindingoctree_add = "/.../masif/nonBS_octree_cg/"
candidate_number= 11
lr=0.001
device="cuda"
checkpoint_path = None
train_model(model_type,epoches,n_fold,bindingoctree_add,nonbindingoctree_add,lr,candidate_number,checkpoint_path,device)

