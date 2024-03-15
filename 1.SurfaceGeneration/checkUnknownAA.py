#!/usr/bin/env python
# coding: utf-8

# In[4]:


import Bio
import sys
from Bio.PDB import *
from dictionary import *
#This is used to check if there are unknown AA
#("enter the code of the pdb(e.g. 8acy):")
protein = sys.argv[1]#code of pdb in lowercase
#print("enter the file address of the pdb(e.g. /home/pdb1a23.ent):")
address = sys.argv[2]#folder address like "/home/pdb1a23.ent"

# parser a protein
struc_dict = PDBParser(QUIET=True).get_structure(protein,address)

#get all atoms info with non-standard
atoms = Selection.unfold_entities(struc_dict, "A")     

c_atoms=list(set([item.get_parent().get_resname() for item in atoms]))

c_atoms=[item for item in c_atoms if not(item in NON_POLYMER) and not(item in k)]
if(len(c_atoms))==0:
    print("There is no unknown AA, go to the next step")
else:
    print("There are unknown AAs, please editting them in dictionary.py")
    print("These unknown AAs are:")
    for a in c_atoms:
        print(a)

