import pandas as pd
import os
import numpy as np
from Bio.PDB import *
import pandas as pd
from tqdm import tqdm

def atom_key(atom):
    res = atom.get_parent()            # Residue
    chain = res.get_parent()           # Chain
    model = chain.get_parent()         # Model
    hetflag, resseq, icode = res.get_id()  # (' ', seq, icode)
    return (
        model.get_id(),
        chain.get_id(),
        hetflag,
        int(resseq),
        icode,
        atom.get_name().strip(),
    )

def write_all_atoms_labeled(all_atoms, pred, true, out_csv="atoms_labeled.csv"):
    # Build key sets for membership tests
    pred_keys = {atom_key(a) for a in pred}
    true_keys = {atom_key(a) for a in true}

    rows = []
    for a in all_atoms:
        k = atom_key(a)
        in_pred = k in pred_keys
        in_true = k in true_keys

        if in_pred and in_true:
            label = 2.0          # both in true and pred
        elif in_true:
            label = 1.0          # only in true
        elif in_pred:
            label = 3.5          # only in pred
        else:
            label = 4.0          # only in all_atoms (neither pred nor true)

        x, y, z = map(float, a.get_coord())
        rows.append((x, y, z, label))

    df = pd.DataFrame(rows, columns=["x", "y", "z", "label"])
    df.to_csv(out_csv, index=False)
    return df
    
true = list(PDBParser(QUIET=True).get_structure("a", "prot_3btsB.pdb_0.pdb" ).get_atoms())
all_atoms = list(PDBParser(QUIET=True).get_structure("a", "prot_3btsB.pdb" ).get_atoms())
pred_df = pd.read_csv("prot_3btsB.pdb_predictions.csv", skipinitialspace=True)
pred_df.columns = pred_df.columns.str.strip()
pockets = dict()
for i, row in pred_df.iterrows():
    if i <= 0:
        pocket_id = int(row["name"].strip().replace('pocket', ''))
        atom_ids = list(map(int, str(row["surf_atom_ids"]).split()))
        atoms = [all_atoms[idx] for idx in atom_ids if idx < len(all_atoms)]
        pockets[pocket_id] = atoms

pred =  pockets[1]

df = write_all_atoms_labeled(all_atoms, pred, true, out_csv="3btsB.csv")

