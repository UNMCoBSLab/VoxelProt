import csv
import os 
import zipfile
from Bio.PDB import *
from VoxelProt.dataset.dictionary import *
from Bio.PDB import PDBList,PDBParser, PDBIO, Select

from collections import defaultdict
class ProteinSelect(Select):
    def accept_residue(self, residue):
        # only standard ATOM residues (hetero flag == ' ')
        return residue.id[0] == ' '

class LigandSelect(Select):
    def accept_residue(self, residue):
        # only HETATM (hetero flag != ' ')  
        # but skip water (HOH)
        return residue.id[0] != ' ' 

class KeepChainsNoWater(Select):
    def __init__(self, keep_chains):
        self.keep = set(keep_chains)
    def accept_chain(self, chain):
        return chain.id in self.keep
    def accept_residue(self, residue):
        # drop waters
        return residue.get_resname() != "HOH"
        

def check_unknown_AA(src_dir):
    parser = PDBParser(QUIET=True)
    num = 0
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".pdb"):
            continue
    
        path = os.path.join(src_dir, fname)
        struct = parser.get_structure(fname, path)
        # gather all atom-parent residue names
        atoms = Selection.unfold_entities(struct, "A")
        c_atoms=list(set([item.get_parent().get_resname() for item in atoms]))
        
        c_atoms=[item for item in c_atoms if not(item in NON_POLYMER) and not(item in k)]
        if len(c_atoms)!=0:
            num = num + 1
            print("There are unknown AAs, please editting them in dictionary.py")
            print("These unknown AAs are:")
            for a in c_atoms:
                print(a)    
    if num ==0:
        print("There is no unknown AA, go to the next step")

        
def split_prot_ligand(src_dir):
    prot_dir = os.path.join(os.path.dirname(src_dir), "split_proteins")
    lig_dir  = os.path.join(os.path.dirname(src_dir), "split_ligands")    
    os.makedirs(prot_dir, exist_ok=True)
    os.makedirs(lig_dir,  exist_ok=True)
    
    parser = PDBParser(QUIET=True)
    io     = PDBIO()

    for fname in os.listdir(src_dir):
        if not fname.endswith(".pdb"):
            continue
        pdb_path = os.path.join(src_dir, fname)
        struct   = parser.get_structure(fname[:-4], pdb_path)
    
        # write protein-only file
        out_prot = os.path.join(prot_dir, "prot_"+fname)
        io.set_structure(struct)
        io.save(out_prot, ProteinSelect())
    
        # write ligand-only file
        lig_res = [r for r in struct.get_residues() if LigandSelect().accept_residue(r)]
        if lig_res:
            out_lig = os.path.join(lig_dir, "lig_"+fname)
            io.set_structure(struct)
            io.save(out_lig, LigandSelect())
        else:
            print(f"No ligands in {fname}, skipping ligand output.")
            
def get_protein_id():
    ids = []
    file_dir = os.path.join(os.getcwd(), "VoxelProt", "dataset", "pdb_list_experiments.csv")
    with open(file_dir, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)            
        for row in reader:
            full = row[0]               
            code, _ = full.split('_', 1) 
            ids.append(code)
    
    return ids    


def download_pdbs(pdb_ids,download_dir):    
    pdbl    = PDBList()
    for pid in pdb_ids:
        pdbl.retrieve_pdb_file(pid,
                               pdir       = download_dir,
                               file_format= "pdb")


def reduce(download_dir):
    print("Run the following commands in terminal:")
    print(f"cd {download_dir}")
    print("mkdir -p stripped withH")          
    print("for f in *.pdb *.ent; do")
    print('[ -e "$f" ] || continue')
    print('base="${f%.*}"')
    print('reduce -Trim "$f" > stripped/"$base".pdb')
    print('reduce -build stripped/"$base".pdb > withH/"${base}"_H.pdb')
    print("done")


def output_pdb_contain_selected_chain(dst_dir,src_dir):
    csv_file = os.path.join(os.getcwd(), "VoxelProt", "dataset", "pdb_list.csv")
    os.makedirs(dst_dir, exist_ok=True)

    parser = PDBParser(QUIET=True)
    io     = PDBIO()

    with open(csv_file) as f:
        for row in csv.reader(f):
            entry = row[0].strip()
            if not entry or "_" not in entry:
                continue
            pdb_id, chain_str = entry.split("_", 1)
            pdb_id_low = pdb_id.lower()
            infile  = os.path.join(src_dir, f"pdb{pdb_id_low}_H.pdb")
            if not os.path.isfile(infile):
                print(f" Missing {infile}")
                continue
    
            # load structure
            struct = parser.get_structure(pdb_id, infile)
            models = list(struct)
            for m in models[1:]:  # skip the very first model
                struct.detach_child(m.id)
                
            
    
            # build Select that keeps exactly the chains listed
            chains_to_keep = list(chain_str)   
            selector = KeepChainsNoWater(chains_to_keep)
    
            # write out
            io.set_structure(struct)
            outfile = os.path.join(dst_dir, f"{pdb_id}_{chain_str}.pdb")
            io.save(outfile, select=selector)
            
    
def run_pdb2pqr(src_dir, dst_dir):
    print("Run the following commands in terminal:")
    print("set -uuo pipefail")
    print()
    print(f'export SRC_DIR="{src_dir}"')
    print(f'export DST_DIR="{dst_dir}"')
    print()
    print('mkdir -p "$DST_DIR"')
    print()
    print('for pdb in "$SRC_DIR"/*.pdb; do')
    print('    base=$(basename "$pdb" .pdb)')
    print('    pqr="$DST_DIR/${base}.pqr"')
    print('    inz="$DST_DIR/${base}.in"')
    print()
    print('    if ! pdb2pqr \\')
    print('        --ff AMBER \\')
    print('        --with-ph 7.0 \\')
    print('        --apbs-input "$inz" \\')
    print('        "$pdb" "$pqr"; then')
    print('        continue')
    print('    fi')
    print('done')
def run_apbs(in_dir, out_dir):
    print("Run the following commands in terminal:")
    print("set -uo pipefail")
    print()
    print(f'export IN_DIR="{in_dir}"')
    print(f'export OUT_DIR="{out_dir}"')
    print('mkdir -p "$OUT_DIR"')
    print()
    # the main loop
    print('for in_file in "$IN_DIR"/*.in; do')
    print('    base=$(basename "$in_file" .in)')
    print()
    print('    (')
    print('      cd "$IN_DIR"')
    print('      apbs "${base}.in"')
    print('    )')
    print('    status=$?')
    print()
    print('    if [[ $status -ne 0 ]]; then')
    print('      continue')
    print('    fi')
    print()
    print('    if [[ -f "$IN_DIR/${base}.dx" ]]; then')
    print('      mv "$IN_DIR/${base}.dx" "$OUT_DIR"/')
    print('    fi')
    print('done')
    print(f'mkdir -p {out_dir}/')
    print(f'mv {in_dir}/*.dx {out_dir}/')
