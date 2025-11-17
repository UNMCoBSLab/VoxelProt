from VoxelProt.detection.detection_helper import * 
model_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/results/resnet/fold0/ResNet_5_0.pth"
device = "cuda"
n_fold = 0
src_dir = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/"  #variables about atom and feature info
dict_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/prob_dict/"  #store down_sampled_index_2_surPts_atomId
#true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/true_binding_site/"
out_fn = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/results/voxelprot/voxelprot_output"
"""
#if doing the coach_cofactor
from VoxelProt.detection.detection_helper import * 
model_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/voxelprot/results/resnet/fold0/ResNet_5_0.pth"
device = "cuda"
n_fold = 0
src_dir = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach_cofactor/" #variables about atom and feature info
dict_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach_cofactor/prob_dict/"  #store down_sampled_index_2_surPts_atomId
#true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach_cofactor/true_binding_site/"
out_fn = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach_cofactor/results/voxelprot_output"

#==========get model and test_list
model = load_model("resnet",model_path,device)
test_list = get_test_id(n_fold,"coach_cofactor")  # or chen_cofactor

"""
#==========get model and test_list
model = load_model("resnet",model_path,device)
test_list = get_test_id(n_fold)

for protein_id in tqdm(test_list[150:200]):
    print(protein_id)
    #==========get all Atom and feature info     
    protein_atoms, surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,si3 = load_atom_feature(protein_id,src_dir)
    #==========get the center of clusters and how many surface points in each cluster 
    sorted_nums, sorted_centroids = get_cluster_centroids(surface_points,si3,eps=1.0,min_samples=100)
    
    #==========downsample surface points    
    if not os.path.exists(f"{dict_path}{protein_id[0]}.pkl"):
        down_sampled_index_2_surPts_atomId = down_sample(protein_atoms, surface_points)  #key: index value(coor, the nearest atom, type, prob)
    else:
        down_sampled_index_2_surPts_atomId = load_dict(f"{dict_path}{protein_id[0]}.pkl")
    
    down_sp =np.array([each[0] for each in down_sampled_index_2_surPts_atomId.values()])
    point_tree=cKDTree(np.array(down_sp))
    
    #==========searching---------phrase I    
    prob_cut_off = 1.0
    down_sampled_index_2_surPts_atomId,sorted_centroids,prob_cut_off,output = search(prob_cut_off,sorted_centroids,model, point_tree, \
           down_sampled_index_2_surPts_atomId,surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype)
    
   
    #==========searching---------phrase II        
    prob_cut_off = prob_cut_off - 0.1
    down_sampled_index_2_surPts_atomId,sorted_centroids,prob_cut_off,output = search(prob_cut_off,sorted_centroids,model, point_tree, \
           down_sampled_index_2_surPts_atomId,surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype)  

    #=========store results
    out_predict_binding_site(out_fn,protein_id,output)
    store_dict(down_sampled_index_2_surPts_atomId,dict_path,f"{protein_id[0]}.pkl") 
    
    


"""
for the coach420
from VoxelProt.detection.detection_helper import * 
model_path = "ResNet_5_chen11.pth"
device = "cuda"
n_fold = 0
src_dir = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/" #variables about atom and feature info
dict_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/prob_dict/"  #store down_sampled_index_2_surPts_atomId
true_binding_site_pdb_path = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/true_binding_site(all)/"
out_fn = "/media/jingbo/HD1_8TB/voxelprot_cofactor/coach420/results/voxelprot/voxelprot_output(all)"

#==========get model and test_list
model = load_model("resnet",model_path,device)
test_list = get_test_id(n_fold,"coach42_all")  # or chen_cofactor, coach42_all,coach42_excluded
for protein_id in tqdm(test_list[50:100]):
    print(protein_id)
    #==========get all Atom and feature info     
    protein_atoms, surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,si3 = load_atom_feature(protein_id,src_dir)
    #==========get the center of clusters and how many surface points in each cluster 
    sorted_nums, sorted_centroids = get_cluster_centroids(surface_points,si3,eps=1.0,min_samples=100)
    
    #==========downsample surface points    
    if not os.path.exists(f"{dict_path}{protein_id}.pkl"):
        down_sampled_index_2_surPts_atomId = down_sample(protein_atoms, surface_points)  #key: index value(coor, the nearest atom, type, prob)
    else:
        down_sampled_index_2_surPts_atomId = load_dict(f"{dict_path}{protein_id}.pkl")
    
    down_sp =np.array([each[0] for each in down_sampled_index_2_surPts_atomId.values()])
    point_tree=cKDTree(np.array(down_sp))
    
    #==========searching---------phrase I    
    prob_cut_off = 1.0
    down_sampled_index_2_surPts_atomId,sorted_centroids,prob_cut_off,output = search(prob_cut_off,sorted_centroids,model, point_tree, \
           down_sampled_index_2_surPts_atomId,surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype)
    
   
    #==========searching---------phrase II        
    prob_cut_off = prob_cut_off - 0.1
    down_sampled_index_2_surPts_atomId,sorted_centroids,prob_cut_off,output = search(prob_cut_off,sorted_centroids,model, point_tree, \
           down_sampled_index_2_surPts_atomId,surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype)  

    #=========store results
    out_predict_binding_site(out_fn,protein_id,output)
    store_dict(down_sampled_index_2_surPts_atomId,dict_path,f"{protein_id}.pkl") 
"""
