#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Bio
from Bio.PDB import *
from dictionary import *
import numpy as np
from surfaceSES import *
from helpers import *
import torch
from atomtype import *
from electrostatics import *
from hydrogen_bond_potential import *
from hydropathy_info import *
from gridData import Grid
from curvatures import *
#This is used to check if there are unknown AA

print("enter the code of the pdb(e.g. 8acy):")
protein = input()
print("enter the file address of the pdb (e.g. /home/pdb1a23.ent):")
pdbaddress = input()#folder address where pdb is stored
print("enter the chain ID (e.g.:ABCD)")
chainid = input()# the chain id that is used to generate the surface
print("enter the file address of the output folder (e.g. /home/features/)")
address=input()#folder address for the output files
print("enter the address of .dx file (e.g. g=Grid(/home/1a23.dx)")
dxaddress=input()
"""
protein='4z9d'
pdbaddress="/home/llab/Desktop/JBLab/detection/MasifDataset/ent/pdb4z9d.ent"
chainid="A" 
address="/home/llab/Desktop/JBLab/detection/MasifDataset/features/"
dxaddress="/home/llab/Desktop/JBLab/detection/MasifDataset/dx/4z9d.dx"
"""
# parser a protein
struc_dict = PDBParser(QUIET=True).get_structure(protein,pdbaddress)
#get all atoms info with non-standard
atoms = Selection.unfold_entities(struc_dict, "A") 
#get the info of protein_atoms        
protein_atoms=[item for item in atoms if (item.get_full_id()[2] in chainid) and (item.get_parent().get_resname() in k)]
protein_atoms_coor=np.array([item.get_coord() for item in protein_atoms])
protein_atoms_res=[item.get_parent().get_resname() for item in protein_atoms]
#this is a 1D list
atomtypes_protein = cal_atomtype_radius(protein_atoms)
# using the protein_atoms to create the surface model
surfacePoints,candidate_index,shape_index,oriented_nor_vector,shape_index3=createSurface    (protein_atoms_coor,atomtypes_protein,sup_sampling=5,scales=[1.5],batch=None)

#surfaceNormals,candidates,shape_index,shape_index3
feathers2CSV(torch.cat((surfacePoints, oriented_nor_vector), 1).cpu(),address+"surfaceNormals/",protein)
feathers2CSV(candidate_index.cpu(),address+"candidates/",protein)
feathers2CSV(shape_index.cpu(),address+"shape_index/",protein)  
feathers2CSV(shape_index3.cpu(),address+"shape_index3/",protein)

protein_atoms_coor=torch.tensor(protein_atoms_coor)
#atomtypes
atomtype_surface=getAtomType(protein_atoms, protein_atoms_res,protein_atoms_coor,surfacePoints.cpu())
feathers2CSV(atomtype_surface,address+"atomtype/",protein)
#KH
KD=getKD(protein_atoms_coor,protein_atoms_res,surfacePoints.cpu())
Hpotential=computeCharge(surfacePoints.cpu(), protein_atoms_coor, protein_atoms)
KD_Hpotential=torch.cat((KD, Hpotential), 1)
feathers2CSV(KD_Hpotential,address+"KH/",protein)
        
g=Grid(dxaddress)
elec=getPoiBol(g,surfacePoints.cpu()) 
feathers2CSV(elec,address+"electro_info/",protein)

