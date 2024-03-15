#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from positiveOctreeNoEnergy import *
from negativeOctreeNoEnergy import *
#this is used to generate the binding site Octrees
protein=sys.argv[1]
pdbAddress=sys.argv[2]
featureaddress=sys.argv[3]
bindingoctreeadd=sys.argv[4]
nonbingdingoctreeadd=sys.argv[5]

numberSelectCandidates=15
device="cuda"
"""
protein="1a27"
pdbAddress="/home/llab/Downloads/1a27.pdb"
featureaddress="/home/llab/Downloads/features/"
bindingoctreeadd="/home/llab/Downloads/BS_octree_N/"
device="cuda"
nonbingdingoctreeadd="/home/llab/Downloads/nonBS_octree_N/"
numberSelectCandidates=15
"""
try:
    #this is used to generate the binding site Octrees   
    positiveOctreeNoEnergy(protein,pdbAddress,featureaddress,octreeadd=bindingoctreeadd,device=device)   
    #this is used to generate the non-binding site Octrees   
    negativeOctreeNoEnergy(protein,pdbAddress,featureaddress,octree_add=nonbingdingoctreeadd,numberSelectCandidates=numberSelectCandidates,device=device)   
    print(protein+" has been successful processed.")
except:
    print("fail to voxelize protein: "+protein)


# In[ ]:




