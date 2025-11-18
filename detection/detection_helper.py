from VoxelProt.trainModel.ResNet import ResNet 

import torch
import torch.nn.functional as F
import os,csv,math,pickle
from tqdm import tqdm
import Bio
from Bio.PDB import *

from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict, OrderedDict

from VoxelProt.surface2Octree.readdata import *
from VoxelProt.surface2Octree.points_SAS import Points
from VoxelProt.surface2Octree.octree import Octree
from VoxelProt.surface2Octree.vdwEnergy import *
from VoxelProt.surface2Octree.dictionary import k

from VoxelProt.evaluation.get_true_binding_site_helper import * 
from VoxelProt.evaluation.eval_helpers import *


        
def out_predict_binding_site(out_fn,protein_id,output):
    if isinstance(protein_id, tuple):
        fn = os.path.join(out_fn, f"{protein_id[0]}_{protein_id[1]}_out")
        if not os.path.exists(fn):
            os.makedirs(fn, exist_ok=True) 
            
        fn_pocket = os.path.join(fn,"pocket")        
        if not os.path.exists(fn_pocket):
            os.makedirs(fn_pocket, exist_ok=True) 

        data = [["Name", "Energy", "Number"]]
        for key,(energy,predicted_binding_site) in output.items():
            atoms_to_pdb(predicted_binding_site, os.path.join(fn_pocket, f"pocket{key+1}.pdb")) 
            data .append([f"Pocket{key+1}",energy,len(predicted_binding_site)])

        with open(os.path.join(fn, f"{protein_id[0]}_{protein_id[1]}_info.txt"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        fn = os.path.join(out_fn, f"{protein_id[5:9].upper()}_{protein_id[9:10]}_out")
        if not os.path.exists(fn):
            os.makedirs(fn, exist_ok=True) 
            
        fn_pocket = os.path.join(fn,"pocket")        
        if not os.path.exists(fn_pocket):
            os.makedirs(fn_pocket, exist_ok=True) 

        data = [["Name", "Energy", "Number"]]
        for key,(energy,predicted_binding_site) in output.items():
            atoms_to_pdb(predicted_binding_site, os.path.join(fn_pocket, f"pocket{key+1}.pdb")) 
            data .append([f"Pocket{key+1}",energy,len(predicted_binding_site)])

        with open(os.path.join(fn, f"{protein_id[5:9].upper()}_{protein_id[9:10]}_info.txt"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)    
        
def compute_probe_energy(pred_binding_site, probe_pos, probe_types = ["C_sp3","O_sp3","H_polar","O_carboxyl","N","C_aromatic"], r_min = 1e-3):   
    rv = 0
    for probe_type in probe_types:
        #print(compute_energy(pred_binding_site, probe_pos, probe_type))
        rv += compute_energy(pred_binding_site, probe_pos, probe_type)
    return rv / len(probe_types)
    
def compute_energy(pred_binding_site, probe_pos, probe_type, r_min = 1e-3):
    """
    Compute E = sum_i [ LJ_ij ] + 
    where LJ_ij = 4 ε_ij [ (σ_ij/r)^12 – (σ_ij/r)^6 ]
          ε_ij = sqrt(ε_i ε_p),  σ_ij = (σ_i + σ_p)/2
    """        
    ke = 332.0636   #Coulomb constant
    eps_0 = 1.0  #dielectric constant
    
    if probe_type == "C_sp3":  #Van der Waals, hydrophobic contacts
        probe_eps = 0.1094
        probe_sigma = 3.399
        probe_charge = 0.0
        
    elif probe_type == "O_sp3": #Accepts H-bonds from donors
        probe_eps = 0.2104
        probe_sigma = 3.000
        probe_charge = -0.66 
        
    elif probe_type == "H_polar":  #Forms H-bonds with acceptors
        probe_eps = 0.0157
        probe_sigma = 2.500
        probe_charge = 0.33
        
    elif probe_type == "O_carboxyl":  #Mimics acidic groups
        probe_eps = 0.2104
        probe_sigma = 3.000
        probe_charge = -1.0  
        
    elif probe_type == "N":  #Mimics NH+
        probe_eps = 0.1700
        probe_sigma = 3.250
        probe_charge = +1.00 
    elif probe_type == "C_aromatic":  
        probe_eps = 0.0860
        probe_sigma = 3.400
        probe_charge =  -0.12
 		
    coords, eps, sigma, charges = load_site_parameters(pred_binding_site)
    
    subcubes = split_cube(probe_pos)
    pseudo_atoms = get_pseudo_atoms(pred_binding_site, subcubes, cutoff=4.0)
    rv = []
    for pseudo_atom in pseudo_atoms:
        diffs = coords - pseudo_atom[None, :]       
        r2    = np.sum(diffs**2, axis=1)         
    
        # Clamp very small distances to r_min^2
        r2 = np.maximum(r2, r_min**2)
        r  = np.sqrt(r2)                          
    
        # Lorentz–Berthelot mixing
        eps_mix   = np.sqrt(eps * probe_eps)      
        sigma_mix = 0.5 * (sigma + probe_sigma)   
    
        # Lennard–Jones 12–6
        sr6 = (sigma_mix / r)**6                 
        e_lj = 4 * eps_mix * (sr6**2 - sr6)   

        
        #rv.append(float(np.sum(e_lj)))  

        # Coulomb energy
        e_coulomb = ke * probe_charge * charges / (eps_0 * r)
        rv.append(float(np.sum(e_lj)) + float(np.sum(e_coulomb))) 
    
    return  np.array(rv).mean()
"""    
def search_pred_binding_site(model,point_tree, detected_center, gap, down_sampled_index_2_surPts_atomId,\
                            surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,prob_cut_off,\
                            device,length=16):
    #------to_visit_spts stores all subsampled surface points with gap from the center
    to_visit_index = point_tree.query_ball_point(detected_center, gap)
    to_visit_spts=point_tree.data[to_visit_index]  #array([[ 19.46391296,   5.15750551, -15.10599136]])
    
    #------search the predicted binding site
    pred_binding_site=[]  
    for j in range(len(to_visit_spts)):
        point = to_visit_spts[j]
        #to shorten the computation time ,0:no 1:yes 2:unvisited; float: this is the probbility  
        #print(down_sampled_index_2_surPts_atomId[to_visit_index[j]])
        if down_sampled_index_2_surPts_atomId[to_visit_index[j]][2]==2:    # unvisited.
            
            
            dataset=octforest_rotated(torch.tensor(point),\
                                      surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,2,device=device,length=16)                          
            (pred_label,pred_prob)=pred_point(model,dataset,batch_size=1,device=device)
            
            down_sampled_index_2_surPts_atomId[to_visit_index[j]] = (down_sampled_index_2_surPts_atomId[to_visit_index[j]][0],\
                                                                     down_sampled_index_2_surPts_atomId[to_visit_index[j]][1],\
                                                                     pred_label,
                                                                     pred_prob
                                                                    )

            
            if pred_label==1 and pred_prob>prob_cut_off:
                pred_binding_site.append(down_sampled_index_2_surPts_atomId[to_visit_index[j]][1])                
             
        elif down_sampled_index_2_surPts_atomId[to_visit_index[j]][2]==1 and down_sampled_index_2_surPts_atomId[to_visit_index[j]][3]>prob_cut_off:
            pred_binding_site.append( down_sampled_index_2_surPts_atomId[to_visit_index[j]][1] )  

    #print("pred_binding_site:",len(pred_binding_site))  
    #ratio = len(pred_binding_site)/len(to_visit_index)
    return pred_binding_site,down_sampled_index_2_surPts_atomId
"""
def search_pred_binding_site(model,point_tree, detected_center, gap, down_sampled_index_2_surPts_atomId,\
                            surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,prob_cut_off,\
                            device,length=16):
    #------to_visit_spts stores all subsampled surface points with gap from the center
    to_visit_index = point_tree.query_ball_point(detected_center, gap)
    to_visit_spts=point_tree.data[to_visit_index]  #array([[ 19.46391296,   5.15750551, -15.10599136]])
    
    #------search the predicted binding site
    pred_binding_site=[]  
    for j in range(len(to_visit_spts)):
        point = to_visit_spts[j]
        #to shorten the computation time ,0:no 1:yes 2:unvisited; float: this is the probbility  
        #print(down_sampled_index_2_surPts_atomId[to_visit_index[j]])
        if down_sampled_index_2_surPts_atomId[to_visit_index[j]][2]==2:    # unvisited.
            
            
            dataset=octforest_rotated(torch.tensor(point),\
                                      surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,2,device=device,length=16)                          
            (pred_label,pred_prob)=pred_point(model,dataset,batch_size=1,device=device)
            
            down_sampled_index_2_surPts_atomId[to_visit_index[j]] = (down_sampled_index_2_surPts_atomId[to_visit_index[j]][0],\
                                                                     down_sampled_index_2_surPts_atomId[to_visit_index[j]][1],\
                                                                     pred_label,
                                                                     pred_prob
                                                                    )

            
            if pred_prob>prob_cut_off:
                pred_binding_site.append(down_sampled_index_2_surPts_atomId[to_visit_index[j]][1])                
             
        elif down_sampled_index_2_surPts_atomId[to_visit_index[j]][3]>prob_cut_off:
            pred_binding_site.append( down_sampled_index_2_surPts_atomId[to_visit_index[j]][1] )  

    #print("pred_binding_site:",len(pred_binding_site))  
    #ratio = len(pred_binding_site)/len(to_visit_index)
    return pred_binding_site,down_sampled_index_2_surPts_atomId
    
    
def get_predicted_output(output, pdb_true, out_fn, protein_id):
    if not os.path.exists(out_fn):
        os.makedirs(out_fn, exist_ok=True) 
        
    min_key = min(output, key=lambda k: output[k][0]) 
    predicted_binding_site = output[min_key][1]
    dvo = compute_dvo(pdb_true, predicted_binding_site)
    dcc = compute_dcc(pdb_true, predicted_binding_site)
    print("DVO: ",dvo)
    print("DCC: ",dcc)     
    atoms_to_pdb(predicted_binding_site, os.path.join(out_fn, f"{protein_id[0]}_{protein_id[1]}.pdb"))    
    return dvo,dcc
    
def store_dict(dict_name,path_name,fn):
    
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True) 
        
    with open(f"{path_name}{fn}", 'wb') as f:
        pickle.dump(dict_name, f)

def load_dict(path_name):
    with open(path_name, 'rb') as f:
        dict_name = pickle.load(f)
    return dict_name


def update_centroids (energy_candidates,old_centroids):

    new_candidates = group_candidates_by_min_energy(energy_candidates)
    new_sorted_centroids = []
    for key, value in new_candidates.items():
        index = value[0]
        new_sorted_centroids.append(old_centroids[index])
    
    if len(new_sorted_centroids)==0:
        return old_centroids
        
    return new_sorted_centroids
    
def group_candidates_by_min_energy(energy_candidates):
    """
    Given a mapping from candidate → list of energy scores,
    reduces each candidate to its minimum energy, then inverts
    that mapping to group candidates by their min energy.
    """
    energy_candidates = { key: min(value)  for key,value in energy_candidates.items()}
    grouped = defaultdict(list)
    for key, val in energy_candidates.items():
        grouped[val].append(key)
    
    for val in grouped:
        grouped[val].sort()

    return {e: grouped[e] for e in sorted(grouped)}
    
def pred_point(model,dataset_cg,batch_size=1,device="cuda"):
    with torch.no_grad():
        yes=0
        no=0
        softmax=[]
        for inputs in dataset_cg:
                outputs = model(inputs,batch_size=batch_size)
                log_softmax = F.log_softmax(outputs,dim=1)     
                pred = torch.argmax(outputs, dim=1)
    
                if (pred==0).item():             
                    no=no+1
                    
                else:
                    yes=yes+1  
                    softmax.append(math.exp(log_softmax[0][1]))
        if yes>=no:
            pred=torch.tensor([1]).to(device)
        else:
            pred=torch.tensor([0]).to(device) 
        try:
            value= max(softmax)
        except:
            value=0

        return (pred,value)

        
def octforest_rotated(center,surfacePoints,oriented_nor_vector,KD_Hpotential,elec,\
                      atomtype,label,device=torch.device("cuda"),length=16):
    rv=[]
    clouds=Points(center,surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),\
                  torch.tensor([label]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=length)
                   
    
    for i in range(4):
        vars()["x"+str(i)]= Octree(depth=4,device = device)
        vars()["x"+str(i)].build_octree(clouds)
        vars()["x"+str(i)].build_neigh()
        rv.append(vars()["x"+str(i)]) 
        clouds.rotate(90,"x")

    for i in range(4,7):
        clouds.rotate(90,"y")
        vars()["x"+str(i)]= Octree(depth=4,device = device)
        vars()["x"+str(i)].build_octree(clouds)
        vars()["x"+str(i)].build_neigh()
        rv.append(vars()["x"+str(i)]) 
    clouds.rotate(90,"y")           

    for i in range(7,10):        
        clouds.rotate(90,"z")
        vars()["x"+str(i)]= Octree(depth=4,device = device)
        vars()["x"+str(i)].build_octree(clouds)
        vars()["x"+str(i)].build_neigh()
        rv.append(vars()["x"+str(i)])  
    return rv
def read_coach420_list(csv_type):
    if csv_type=="coach420_all":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_all-prt2lig.csv")
    elif csv_type=="coach420_excluded":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_excluded-prt2lig.csv")  
        
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        rv = []
        for protein, lig_str in reader:
            rv.append(protein)
    return rv
    
def read_holo_list():
    pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "HOLO4K-prot2lig.csv")  
        
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        rv = []
        for protein, lig_str in reader:
            rv.append(protein)
    return rv
    
def get_test_id(n_fold,csv_type = "masif_data"):
    if csv_type == "HOLO4K":
        return read_holo_list()
    if csv_type=="coach420_all" or csv_type=="coach420_excluded":
        return read_coach420_list(csv_type)
    ids = []
    if csv_type == "masif_data":
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "cross_val_splits",f"fold_{n_fold}","test.txt")
    elif csv_type == "coach_cofactor":
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_cofactor.csv")
    elif csv_type == "chen_cofactor":
        file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "chen_cofactor.csv")
    with open(file_dir, newline='') as f:
        reader = csv.reader(f)          
        for row in reader:
            full = row[0]               
            code, chain = full.split('_', 1) 
            ids.append((code,chain))
    
    return ids 
    
def collapse_and_sort_by_energy(output_dict):
    """
    Given a dict mapping arbitrary keys to (score, atoms) tuples,
    collapse entries with identical scores (keeping the first one),
    sort by score ascending, and re-index from 0 upwards.
    """

    seen = {}
    for _, (score, atoms) in output_dict.items():
        s = float(score)
        if s not in seen:
            seen[s] = atoms
    sorted_scores = sorted(seen.items(), key=lambda kv: kv[0])

    return {new_idx: (score, atoms)for new_idx, (score, atoms) in enumerate(sorted_scores)}

def load_model(model_type,model_path,optimizer=None, device='cuda'):
    try:
        model=ResNet(channel_in=16,num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True)) 
    except:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])   
    return model.to(device)

def load_atom_feature(protein_id,src_dir):
    try:
        struc_dict = PDBParser(QUIET=True).get_structure(protein_id, os.path.join(src_dir, "split_proteins",f"prot_{protein_id[0]}_{protein_id[1]}.pdb"))
        add_Surface = os.path.join(src_dir, "features","surfaceNormals",f"{protein_id[0]}.csv")
        add_KD = os.path.join(src_dir, "features","KH",f"{protein_id[0]}.csv")
        add_Elec = os.path.join(src_dir, "features","electro_info",f"{protein_id[0]}.csv")
        add_Candidates = os.path.join(src_dir, "features","candidates",f"{protein_id[0]}.csv")
        add_Atom = os.path.join(src_dir, "features","atomtype",f"{protein_id[0]}.csv")
        add_si3 = os.path.join(src_dir, "features","shape_index3",f"{protein_id[0]}.csv")
        
    except:
        struc_dict = PDBParser(QUIET=True).get_structure(protein_id, os.path.join(src_dir, "split_proteins",protein_id))
        add_Surface = os.path.join(src_dir, "features","surfaceNormals",f"{protein_id}.csv")
        add_KD = os.path.join(src_dir, "features","KH",f"{protein_id}.csv")
        add_Elec = os.path.join(src_dir, "features","electro_info",f"{protein_id}.csv")
        add_Candidates = os.path.join(src_dir, "features","candidates",f"{protein_id}.csv")
        add_Atom = os.path.join(src_dir, "features","atomtype",f"{protein_id}.csv")
        add_si3 = os.path.join(src_dir, "features","shape_index3",f"{protein_id}.csv")
        
    atoms = Selection.unfold_entities(struc_dict, "A")   
    protein_atoms=[item for item in atoms if item.get_parent().get_resname() in k]  # a list of Atom
        
    surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,_= readData(add_Surface,add_KD,add_Elec,add_Candidates,add_Atom)  
    si3=readSI(add_si3) #get the SI when scale=[3.0] 
    return protein_atoms, surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,si3

def get_cluster_centroids(surfacePoints,si3,eps=1.0,min_samples=100):
    #based on the SI_3 to get all clusters    S    
    X = surfacePoints[si3.reshape(1,si3.shape[0])[0]<=0].cpu().numpy() #X represents the surface points with SI<=0.0
    #create teh DBSCAN
    db=DBSCAN(eps = eps, min_samples = min_samples).fit_predict(X)
    #get all surface points within each cluster
    labels = sorted(set(db) - {-1})
    clusters = [ torch.tensor(X[db == lbl], dtype=torch.float32) for lbl in labels]
    
    centroids = []
    num_in_cluster = []
    for i, pts in enumerate(clusters):
        centroid = pts.mean(dim=0)
        centroids.append(centroid.tolist())
        num_in_cluster.append(pts.shape[0])
    pairs = sorted(zip(num_in_cluster, centroids), key=lambda x: x[0], reverse=True)
    sorted_nums, sorted_centroids = map(list, zip(*pairs))
    return sorted_nums, sorted_centroids
def down_sample(protein_atoms, surface_points, cutoff = 3.0):    
    """
    For each surface point, find its nearest protein atom within cutoff
    then for each atom, the single closest surface point.

    Returns:
        dict mapping  index -> (down_sampled_surface_points, Atom, 2, 0.0), 
        where 0:no 1:yes 2:unvisited; float: this is the probbility
    """
    
    atom_coords = np.vstack([atom.get_coord() for atom in protein_atoms])  
    surf_pts    = np.asarray(surface_points) 
    tree = cKDTree(atom_coords) 
    closest_distances, indices = tree.query(surf_pts)
    atomId_surPts_dict = dict()
    for i in range(len(indices)):
        index = indices[i]  # like 135
        atomId = protein_atoms[index]  # like CA
    
        dist = closest_distances[i]  # like 1.60350084
    
        surface_point = surf_pts[i]
        if dist > cutoff:
            continue
            
        if not (atomId in atomId_surPts_dict):
             atomId_surPts_dict[atomId] = (surface_point,dist)
        else:
            if atomId_surPts_dict[atomId][1] > dist:
                atomId_surPts_dict[atomId] = (surface_point,dist) 
    index_surPts_atomId = dict()
    for i, (atom, (coord_array, distance)) in enumerate(atomId_surPts_dict.items()):
        index_surPts_atomId[i] = (tuple(coord_array.tolist()), atom, 2, 0)

    return index_surPts_atomId     

def get_binding_site_number(target_protein_id):
    file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "bindingSiteNumber.csv")
    with open(file_dir, newline='') as f:
        reader = csv.reader(f)     
        for row in reader:
            if row[0] == target_protein_id:
                number = row[1]
                return int(number )
        return -1    

def update_detected_center (pred_binding_site):
    """return the new detected_center and center_pre
    """
    new_detected_center = np.vstack([atom.get_coord() for atom in pred_binding_site])

    return new_detected_center.mean(axis=0)  


def load_site_parameters(pred_binding_site):
    """Parse PDB, return arrays of coords, eps, sigma"""
    atoms     = pred_binding_site

    coords = np.array([a.get_coord() for a in atoms])
    eps    = np.zeros(len(atoms))
    sigma  = np.zeros(len(atoms))
    charge = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        atom_type = atom2vdwatom[(atom.get_parent() .get_resname() ,atom.get_name()) ]
        if atom_type in ff18SB:
            sigma[i], eps[i] = ff18SB[atom_type]
        else:
            sigma[i], eps[i] = (0.1, 3.5)

        try:
            chg = charge_ff19SB[(atom.get_parent() .get_resname() ,atom.get_name())]
        except:
            chg = 0
            
        charge[i] = chg
        
    return coords, eps, sigma, charge

def split_cube(center):
    """ Given a center (x, y, z), creates a cube of side 16 Å
    centered at that point, divides it into divisions^3 equal subcubes,
    and returns a list of the 8 corner coordinates for each subcube.
    """
    length=16.0
    divisions=4
    
    cx, cy, cz = center
    half = length / 2.0
    step = length / divisions

    xs = [cx - half + i*step for i in range(divisions+1)]
    ys = [cy - half + i*step for i in range(divisions+1)]
    zs = [cz - half + i*step for i in range(divisions+1)]

    subcubes = []
    for i in range(divisions):
        for j in range(divisions):
            for k in range(divisions):
                x0, x1 = xs[i], xs[i+1]
                y0, y1 = ys[j], ys[j+1]
                z0, z1 = zs[k], zs[k+1]

                corners = [(x0, y0, z0),(x1, y0, z0),(x0, y1, z0),(x0, y0, z1),(x1, y1, z0),(x1, y0, z1),(x0, y1, z1),(x1, y1, z1)]
                subcubes.append(corners)
    flat = [corner for cube in subcubes for corner in cube]

    return np.array(flat) # (512,3)
    
def get_pseudo_atoms(atoms, points, cutoff=4.0):
    """Given a list of Atom and a list of 3D points, eturn only those points that lie at least 4 Å away from every atom.
    """
    atom_coords = np.array([atom.get_coord() for atom in atoms]) 
    pts = np.asarray(points)
    tree = cKDTree(atom_coords)
    
    distances, _ = tree.query(pts, k=1)    
    # Keep only those points whose nearest‐atom distance >= cutoff
    mask = distances >= cutoff
    pseudo_atoms = pts[mask]    
    return pseudo_atoms
    

def search(prob_cut_off,sorted_centroids,model, point_tree, \
           down_sampled_index_2_surPts_atomId,surface_points,oriented_nor_vector,KD_Hpotential,elec,\
           atomtype,detection_radias = 15):
    output = dict()   #a dictionary,{index of possible binding site: (energy,[binding site atoms])}    
    energy_candidates = dict()   #a dictionary,{index of possible binding site: [energy for each gap]    
    while len(output)==0 :  
        prob_cut_off = prob_cut_off - 0.01
        #print('prob_cut_off',prob_cut_off)
        for ind in range(len(sorted_centroids)):
            # initialize 
            gap, detected_center, center_pre = 3, sorted_centroids[ind],np.zeros(3)   
            while gap < detection_radias: # keep searching the binding site, until meets the stop critirion    
                 #------search the predicted binding site
                pred_binding_site, down_sampled_index_2_surPts_atomId = search_pred_binding_site(model, point_tree, detected_center, gap, down_sampled_index_2_surPts_atomId,\
                                    surface_points,oriented_nor_vector,KD_Hpotential,elec,atomtype,prob_cut_off,\
                                    device="cuda",length=16)
                
                #decided if stop the searching   
                #print('pred_binding_site',len(pred_binding_site))
                if len(pred_binding_site)==0: break   

                #update the new center                
                center_pre=detected_center 
                detected_center = update_detected_center (pred_binding_site)
        
                    
                #update to the next gap
                if compute_euc_dist(detected_center,center_pre)<0.1: 
                    if gap >=9:    
                        #evaluation within this probe
                        energy = compute_probe_energy(pred_binding_site, detected_center)
                        if ind in energy_candidates:                
                            energy_candidates[ind].append(energy)
                        else:
                            energy_candidates[ind] = []
                            energy_candidates[ind].append(energy)
                        
                        if ind in output:
                            if output[ind][0] > energy:
                                output[ind] = (energy,pred_binding_site)# get the energy
                        else:
                            output[ind] = (energy,pred_binding_site)
                    gap=gap+1
                    
    sorted_centroids = update_centroids (energy_candidates,sorted_centroids)                            
    return down_sampled_index_2_surPts_atomId,sorted_centroids,prob_cut_off,collapse_and_sort_by_energy(output)                     

