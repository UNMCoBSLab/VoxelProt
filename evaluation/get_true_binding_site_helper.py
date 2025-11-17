import os,csv
from VoxelProt.surface2Octree.dictionary import *
import Bio
from Bio.PDB import *
from VoxelProt.surface2Octree.readdata import *
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cdist
import networkx as nx

def write_one_bs(tree,bpts,filtered_atoms):
    closest_distances, indices = tree.query(bpts)
    closest_atoms     = [filtered_atoms[i] for i in indices]
    return set(closest_atoms)

def write_bs_coach(fn_p,bpts,out_fn_path,protein_id,ind):
    if not os.path.exists(out_fn_path):
        os.makedirs(out_fn_path, exist_ok=True)

    structure_p = PDBParser(QUIET=True).get_structure("p", fn_p)
    all_atoms   = Selection.unfold_entities(structure_p, "A")
    filtered_atoms = [ atom for atom in all_atoms if atom.get_parent().get_resname() in k]
    
    # build & query the KD-tree
    atom_coords = np.vstack([atom.get_coord() for atom in filtered_atoms])
    tree = cKDTree(atom_coords) 

    #binding_site = write_one_bs(tree,bpts[0],filtered_atoms)
    binding_site = write_one_bs(tree,bpts,filtered_atoms)
    atoms_to_pdb(binding_site, os.path.join(out_fn_path,f"{protein_id}_{ind}.pdb"))
    print(f"success: {protein_id}_{ind}.pdb")


def get_ligands_dic_coach(fn_l):
    structure_l = PDBParser(QUIET=True) .get_structure("p", fn_l)    
    ligand_dic = {}
    
    for model in structure_l:            
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                resid   = residue.get_id()[1]  

                key = f"{chain.id}_{resid}"    

                if key not in ligand_dic:
                    ligand_dic[key] = []

                for atom in residue:
                    ligand_dic[key].append(atom)   
    
    return ligand_dic

def atoms_to_pdb(atom_list, out_fn):    
    groups = defaultdict(list)
    for atom in atom_list:
        res = atom.get_parent()
        chain_id = res.get_parent().id
        hetfield, resseq, icode = res.get_id()
        key = (chain_id, (hetfield, resseq, icode), res.get_resname())
        groups[key].append(atom)

    # a new Structure object
    struct = Structure.Structure("X")
    model  = Model.Model(0)
    struct.add(model)

    # Chain > Residue > Atoms
    for (chain_id, res_id, resname), atoms in groups.items():
        if not model.has_id(chain_id):
            chain = Chain.Chain(chain_id)
            model.add(chain)
        else:
            chain = model[chain_id]

        # add residue
        residue = Residue.Residue(res_id, resname, segid="")
        chain.add(residue)

        # add the atoms
        for atom in atoms:
            residue.add(atom)

    io = PDBIO()
    io.set_structure(struct)
    io.save(out_fn)
    
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
    
    
def read_holo_list(csv_type):
    if csv_type=="HOLO4K_all":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "HOLO4K_all-prot2lig.csv")  
    else:
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "HOLO4K_excluded-prot2lig.csv")  
        
    with open(pdb_list, newline='') as f:
        reader = csv.reader(f)
        rv = []
        for protein, lig_str in reader:
            rv.append(protein)
    return rv
    
def get_pdb_list(csv_type="masif_data"):
    if csv_type=="HOLO4K_all" or csv_type=="HOLO4K_excluded":
        return read_holo_list(csv_type)
    if csv_type=="coach420_all" or csv_type=="coach420_excluded":
        return read_coach420_list(csv_type)
    if csv_type=="masif_data":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "pdb_list_experiments.csv")
    elif csv_type=="coach_cofactor":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "coach420_cofactor.csv")
    elif csv_type=="chen_cofactor":
        pdb_list = os.path.join(os.getcwd(), "VoxelProt", "dataset", "chen_cofactor.csv")
        
        
    dict_data={}
    with open(pdb_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            dict_data[lines[0].split("_")[0]]=lines[0].split("_")[1]
    return dict_data    

def get_ligands_dic(fn_l):
    structure_l = PDBParser(QUIET=True) .get_structure("p", fn_l)    
    ligand_dic = {}
    
    for model in structure_l:            
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                resid   = residue.get_id()[1]  
    
                if resname in COFACTOR:
                    key = f"{chain.id}_{resid}"  

                    if key not in ligand_dic:
                        ligand_dic[key] = []
    
                    for atom in residue:
                        ligand_dic[key].append(atom)    

    
    return ligand_dic

def get_binding_surface_points(ligand_dic,spts,threshold):
    bindingPoints = []
    for index, (key, cofactor_atoms) in enumerate(ligand_dic.items()): 
        ns = Bio.PDB.NeighborSearch(cofactor_atoms)
        bpts=[item for item in spts if len(ns.search(item, threshold))>0 ]
        bindingPoints.append(bpts)
    return bindingPoints   


def write_bs(fn_p,bpts,out_fn_path,protein_id):
    if not os.path.exists(out_fn_path):
        os.makedirs(out_fn_path, exist_ok=True)

    structure_p = PDBParser(QUIET=True).get_structure("p", fn_p)
    all_atoms   = Selection.unfold_entities(structure_p, "A")
    filtered_atoms = [ atom for atom in all_atoms if atom.get_parent().get_resname() in k]
    # build & query the KD-tree
    atom_coords = np.vstack([atom.get_coord() for atom in filtered_atoms])
    tree = cKDTree(atom_coords) 
    for index in range(len(bpts)):
        binding_site = write_one_bs(tree,bpts[index],filtered_atoms)
        atoms_to_pdb(binding_site, os.path.join(out_fn_path,f"{protein_id}_{index}.pdb"))
        print(f"success: {protein_id}_{index}.pdb")

def get_true_binding_site(protein_dir,lid_dir,out_fn_path,addSurface,csv_type="masif_data",threshold = 4):
    dict_data = get_pdb_list(csv_type)
    if csv_type=="masif_data" or csv_type=="chen_cofactor" or csv_type=="coach_cofactor":
        for protein_id, chain in tqdm(dict_data.items()):
            try:    
                fn_l = lid_dir+f"lig_{protein_id}_{chain}.pdb"
                ligand_dic = get_ligands_dic(fn_l)                  
                spts = readSurfacePts(addSurface+f"{protein_id}.csv")
                
                bpts = get_binding_surface_points(ligand_dic,spts,threshold)
                fn_p = os.path.join(protein_dir, f"prot_{protein_id}_{chain}.pdb")
                write_bs(fn_p,bpts,out_fn_path,protein_id)

            except:
                print(protein_id)
    elif (csv_type=="coach420_all" or csv_type=="coach420_excluded"or csv_type=="HOLO4K_all"or csv_type=="HOLO4K_excluded"):
        for protein_id in dict_data:
            try:
                spts = readSurfacePts(addSurface+f"{protein_id}.csv")
            except:
                print(f"{protein_id} doesn't exit")
                continue  
            for ind in range(50):
                try:
                    fn_l = lid_dir+f"lig_{protein_id[5:10]}_{ind}.pdb"
                    ligand_dic = get_ligands_dic_coach(fn_l)                    
                    bpts = get_binding_surface_points(ligand_dic,spts,threshold)
                    bpts = [pt for group in bpts for pt in group]
                    fn_p = os.path.join(protein_dir, protein_id)
                    write_bs_coach(fn_p,bpts,out_fn_path,protein_id,ind)
                except:
                    pass


def insert_ter_on_residue_change(input_pdb_path: str, output_pdb_path: str):
    input_path = Path(input_pdb_path)
    output_path = Path(output_pdb_path)

    with input_path.open("r") as f:
        lines = f.readlines()

    output_lines = []
    prev_id = None

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            resname = line[17:20].strip()
            chain_id = line[21].strip()
            resnum = line[22:26].strip()
            current_id = (resname, chain_id, resnum)

            if prev_id and current_id != prev_id:
                output_lines.append("TER\n")
            output_lines.append(line)
            prev_id = current_id
        else:
            output_lines.append(line)

    # Add a final TER if the last line is an ATOM/HETATM
    if output_lines and (output_lines[-1].startswith("ATOM") or output_lines[-1].startswith("HETATM")):
        output_lines.append("TER\n")

    with output_path.open("w") as f:
        f.writelines(output_lines)


def extract_ligand_coords(pdb_lines):
    ligands = dict()
    atom_lines = dict()
    for line in pdb_lines:
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            chain = line[21].strip()
            resnum = line[22:26].strip()
            key = (resname, chain, resnum)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if key not in ligands:
                ligands[key] = []
                atom_lines[key] = []
            ligands[key].append([x, y, z])
            atom_lines[key].append(line)
    return {k: np.array(v) for k, v in ligands.items()}, atom_lines


        
def cluster_ligands_by_distance(ligand_coords, exclude_resnames=None, distance_cutoff=4.0):
    if exclude_resnames is None:
        exclude_resnames = {'HOH'}
    ligand_keys = [k for k in ligand_coords if k[0] not in exclude_resnames]
    G = nx.Graph()
    G.add_nodes_from(ligand_keys)
    for i in range(len(ligand_keys)):
        for j in range(i + 1, len(ligand_keys)):
            key1, key2 = ligand_keys[i], ligand_keys[j]
            coords1 = ligand_coords[key1]
            coords2 = ligand_coords[key2]
            min_dist = np.min(cdist(coords1, coords2))
            if min_dist <= distance_cutoff:
                G.add_edge(key1, key2)
    clusters = list(nx.connected_components(G))
    return clusters
   
def rewrite_pdb_by_binding_site(pdb_lines, clusters, atom_lines_dict, output_file, exc=None):
    if exc is None:
        exc = set()

    output_lines = []

    for i, cluster in enumerate(clusters):
        cluster = sorted(cluster)

        # Check if ALL ligands in this cluster are in exclusion list
        all_excluded = all(lig_key[0] in exc for lig_key in cluster)

        if all_excluded: continue  # Skip this cluster

        # Otherwise, write this cluster
        for lig_key in cluster:
            output_lines.extend(atom_lines_dict[lig_key])
        output_lines.append("TER\n")

    # Write to output file
    with open(output_file, "w") as f:
        f.writelines(output_lines)


def split_pdb_by_ter(input_pdb_path, output_dir, filename):
    input_path = Path(input_pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as f:
        lines = f.readlines()

    blocks = []
    current_block = []

    for line in lines:
        current_block.append(line)
        if line.strip().startswith("TER"):
            blocks.append(current_block)
            current_block = []

    if current_block: blocks.append(current_block)

    for i, block in enumerate(blocks):
        out_path = output_dir / f"{filename}_{i}.pdb"
        with out_path.open("w") as f:
            f.writelines(block)
            
def process_ligand(input_dirt, distance_cutoff = 4.0,exc=None,print_out=True):
    parent_dir = os.path.dirname(os.path.dirname(input_dirt))
    output_dirt = os.path.join(parent_dir, "split_ligands_with_TER")
    reclusterd_dirt_all= os.path.join(parent_dir, "split_ligands_reclusterd_TER(all)")
    single_output_all = os.path.join(parent_dir, "split_ligands_single(all)")
    reclusterd_dirt_excluded= os.path.join(parent_dir, "split_ligands_reclusterd_TER(excluded)")
    single_output_excluded = os.path.join(parent_dir, "split_ligands_single(excluded)")
    for path in [input_dirt, output_dirt, reclusterd_dirt_all, single_output_all,reclusterd_dirt_excluded,single_output_excluded]:
        os.makedirs(path, exist_ok=True)
        
    if exc is None: exc = set()

    for each in os.listdir(input_dirt):
        fn_in = os.path.join(input_dirt,each)
        fn_out = os.path.join(output_dirt,each)
        #step 1:insert ter once residue changed
        insert_ter_on_residue_change(fn_in, fn_out)
        
        #step2: get all cluster
        with open(fn_out, "r") as f:
            pdb_lines = f.readlines()
    
        ligand_coords, atom_lines_dict = extract_ligand_coords(pdb_lines)
        #compute_inter_ligand_distances(ligand_coords, distance_cutoff=distance_cutoff)
        clusters = cluster_ligands_by_distance(ligand_coords, distance_cutoff=distance_cutoff)
        if print_out:
            print(f"\nBinding site count: {len(clusters)}")
            for i, cluster in enumerate(clusters, 1):
                print(f"Binding Site {i}:")
                for lig in sorted(cluster):
                    print(f"   {lig}")
                
        #step3: rewrite the ligands based on cluster -->all
        output_file = os.path.join(reclusterd_dirt_all , each)
        rewrite_pdb_by_binding_site(pdb_lines, clusters, atom_lines_dict, output_file, None)
        #step4: split into single ligands
        split_pdb_by_ter(output_file, single_output_all,each[0:9])
        
        #step5: rewrite the ligands based on cluster -->excluded
        output_file = os.path.join(reclusterd_dirt_excluded , each)
        rewrite_pdb_by_binding_site(pdb_lines, clusters, atom_lines_dict, output_file, exc)
        
        #step6: split into single ligands
        split_pdb_by_ter(output_file, single_output_excluded,each[0:9])    
