from VoxelProt.detection.detection_helper import *

model_path = "example/ResNet_5_0.pth"
device = "cuda"
n_fold = 0

src_dir   = "example/"              # atom and feature info
dict_path = "example/prob_dict/"    # down_sampled_index_2_surPts_atomId
out_fn    = "example/voxelprot_output"

# ========== path checks ==========

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.isdir(src_dir):
    raise FileNotFoundError(f"Source folder not found: {src_dir}")

if not os.path.isdir(dict_path):
    os.makedirs(dict_path, exist_ok=True)
    print(f"Created dict folder: {dict_path}")

out_dir = os.path.dirname(out_fn) or "."
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output folder: {out_dir}")

# ========== main code ==========

protein_id = "prot_1bxm"    # base name without .pdb
pdb_name = protein_id + ".pdb"

print("Running detection for:", protein_id)

# get all Atom and feature info
protein_atoms, surface_points, oriented_nor_vector, KD_Hpotential, elec, atomtype, si3 = \
    load_atom_feature(pdb_name, src_dir)

# get cluster centers
sorted_nums, sorted_centroids = get_cluster_centroids(
    surface_points, si3, eps=1.0, min_samples=100
)

# load model
model = load_model("resnet", model_path, device)

# load or compute downsample dict
dict_file = os.path.join(dict_path, f"{protein_id}.pkl")

if not os.path.exists(dict_file):
    down_sampled_index_2_surPts_atomId = down_sample(protein_atoms, surface_points)
else:
    down_sampled_index_2_surPts_atomId = load_dict(dict_file)

down_sp = np.array([each[0] for each in down_sampled_index_2_surPts_atomId.values()])
point_tree = cKDTree(np.array(down_sp))

# searching – phase I
prob_cut_off = 1.0
down_sampled_index_2_surPts_atomId, sorted_centroids, prob_cut_off, output = search(
    prob_cut_off,
    sorted_centroids,
    model,
    point_tree,
    down_sampled_index_2_surPts_atomId,
    surface_points,
    oriented_nor_vector,
    KD_Hpotential,
    elec,
    atomtype,
)

# searching – phase II
prob_cut_off = prob_cut_off - 0.1
down_sampled_index_2_surPts_atomId, sorted_centroids, prob_cut_off, output = search(
    prob_cut_off,
    sorted_centroids,
    model,
    point_tree,
    down_sampled_index_2_surPts_atomId,
    surface_points,
    oriented_nor_vector,
    KD_Hpotential,
    elec,
    atomtype,
)

# save results
out_predict_binding_site(out_fn, protein_id, output)
store_dict(down_sampled_index_2_surPts_atomId, dict_path, f"{protein_id}.pkl")

print(f"Detection done; check outputs under {out_fn}*")

