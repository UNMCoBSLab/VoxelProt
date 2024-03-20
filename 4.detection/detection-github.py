#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Bio
from Bio.PDB import *
from readdata import *
import numpy as np
from dictionary import NON_POLYMER,k,polarHydrogens, donorAtom,acceptorAngleAtom
import pykeops
from vgg import VGG_net 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from points import Points
from octree import Octree
import math
import scipy.spatial as spatial
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import sys


# In[3]:


protein=sys.argv[1]
pdbadd=sys.argv[2]
PATH=sys.argv[3]
direction=sys.argv[4]
outputname=sys.argv[5]

#get the info
addSurface=direction+"/features/surfaceNormals/"+protein+".csv"
addKD=direction+"/features/KH/"+protein+".csv"
addElec=direction+"/features/electro_info/"+protein+".csv"
addCandidates=direction+"/features/candidates/"+protein+".csv"   
addAtom=direction+"/features/atomtype/"+protein+".csv"   

#load the model
device="cuda"
model = VGG_net(channel_in=16).to(device)
model.load_state_dict(torch.load(PATH))
"""
protein="4p68"
direction="/home/llab/Desktop/JBLab/detection/4.all"
pdbadd=direction+"/ent/pdb"+protein+".ent"
#get the info
addSurface=direction+"/features/surfaceNormals/"+protein+".csv"
addKD=direction+"/features/KH/"+protein+".csv"
addElec=direction+"/features/electro_info/"+protein+".csv"
addCandidates=direction+"/features/candidates/"+protein+".csv"   
addAtom=direction+"/features/atomtype/"+protein+".csv"   
#load the model
device="cuda"
model = VGG_net(channel_in=16).to(device)
PATH=direction+"/expriment/fold1/all_16_fold1_0.pth"
model.load_state_dict(torch.load(PATH))
outputname='bindingsite'+protein+'.csv'
"""


# In[14]:


def euc_dist(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)

#evaluation points:
#return the result and the probability 
def evaluation_points3(model,dataset,batch_size=1,device="cuda"):
    yes=0
    no=0
    softmax=[]
    for inputs in dataset:
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
    #value=value.cpu().item()
    return [pred,value]

def cal_DVO(true_sampled_bindingATOM,detected_sampled_bindingATOM ):
    #calculate the union
    true_sampled_bindingATOM=true_sampled_bindingATOM.tolist()
    detected_sampled_bindingATOM=detected_sampled_bindingATOM.tolist()
    union=torch.tensor(true_sampled_bindingATOM+detected_sampled_bindingATOM)
    union=subsample(union,scale=1.0) 
    #calculate the intersection
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(true_sampled_bindingATOM)
    intersection=[pi for pi in detected_sampled_bindingATOM if neigh.kneighbors([pi])[0][0][0]<math.sqrt(2)]
    intersection=torch.tensor(intersection)    
    return intersection.shape[0]/union.shape[0]  

def subsample(x, batch=None, scale=1.0):
    if batch is None:  # Single protein case:
            labels = pykeops.torch.cluster.grid_cluster(x, scale).long()
            C = labels.max() + 1
            # We append a "1" to the input vectors, in order to
            # compute both the numerator and denominator of the "average"
            #  fraction in one pass through the data.
            x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
            D = x_1.shape[1]
            points = torch.zeros_like(x_1[:C])
 
            points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
           
    return (points[:, :-1] / points[:, -1:]).contiguous()
def octforest_rotated(center,surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,label,device=torch.device("cuda"),length=16):
    rv=[]
    clouds=Points(center,surfacePoints,oriented_nor_vector,torch.cat([KD_Hpotential,elec,atomtype],1),torch.tensor([label]*surfacePoints.size(0)).view(surfacePoints.size(0),1),device,length=length)
    for i in range(4):
        vars()["x"+str(i)]= Octree(depth=4,device = device)
        vars()["x"+str(i)].build_octree(clouds)
        vars()["x"+str(i)].build_neigh()
        rv.append(vars()["x"+str(i)]) 
        clouds.rotate(90,"x")

    
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


def evaluation_points1(model,octreeset,batch_size=1,device="cuda"):
    yes=0
    no=0
    #print("+++++++++++++++++")
    for inputs in octreeset:
            outputs = model(inputs,batch_size=batch_size)
            pred = torch.argmax(outputs, dim=1)
            #softmax=math.exp(F.log_softmax(outputs,dim=1)[0][pred])
            if (pred==0):
                no=no+1
                #print("0:",softmax)
            else:
                yes=yes+1   
                #print("1:",softmax)
    if yes>=no:
        pred=torch.tensor([1]).to(device)
    else:
        pred=torch.tensor([0]).to(device)   
    return pred


def findPositiveCandidates(model,candidates,surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,batch_size=1,device="cuda",length=16):
    rv=[]
    for ii in range(len(candidates)):
            octreeset=octforest_rotated(candidates[ii].clone().detach(),surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,2,device=device,length=length)       
            if evaluation_points1(model,octreeset,batch_size=batch_size,device=device)==1:
                rv.append(candidates[ii])
    return rv
def sptoatom(visited_yes,protein_atoms_coor):
    #to get the sampled detected binding ATOM and center
    atoms=protein_atoms_coor.tolist()  
    neigh=NearestNeighbors(n_neighbors=1)
    neigh.fit(atoms)
    
    bindingATOM=[]
    for ind in range(len(visited_yes)):
        a=atoms[neigh.kneighbors([visited_yes[ind]])[1][0][0]]
        if not(a in bindingATOM):
            bindingATOM.append(a)   
    rvATOM=bindingATOM.copy()
    bindingATOM=torch.tensor(bindingATOM) 
    sampled_bindingATOM=subsample(bindingATOM,scale=1.0) 
    x,y,z=0,0,0
    for i in sampled_bindingATOM:
        x=x+i[0].item()
        y=y+i[1].item()
        z=z+i[2].item()
    center=[x/len(sampled_bindingATOM),y/len(sampled_bindingATOM),z/len(sampled_bindingATOM)]
    
    return rvATOM,sampled_bindingATOM,center 
def writeOutput(outputname,inputlist):
    with open(outputname,'w') as f:
        writer = csv.writer(f  ,delimiter=',')
        writer.writerows(inputlist)


# In[4]:


# parser a protein
struc_dict = PDBParser(QUIET=True).get_structure(protein, pdbadd)
#get all atoms info with non-standard
atoms = Selection.unfold_entities(struc_dict, "A")    

protein_atoms=[item for item in atoms if item.get_parent().get_resname() in k]    

#get the residue name of protein_atoms
protein_atoms_res=[item.get_parent().get_resname() for item in protein_atoms]

#get the corr info of protein_atoms
protein_atoms_coor=torch.tensor(np.array([item.get_coord() for item in protein_atoms]))        
surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates=readData(addSurface,addKD,addElec,addCandidates,addAtom)      
#----------------------------------
sp=subsample(surfacePoints,scale=1.0)       
sp=sp.tolist()
# create a dict
#key: the coor of each sp
#value:(int,float);1:yes 0:no 2:unvisited;float: this is the probbility
sp_dict={}
for p in sp:
    sp_dict[tuple(p)]= (2,0)  
#----------------------------------
#finding out all positive candidates
positive_candidates=findPositiveCandidates(model,candidates,surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,batch_size=1,device=device,length=16)
point_tree=spatial.cKDTree(np.array(sp))
#create a dict for ratio
#key: (c,r)
#value:filter_ratio
radio_dict={}
#create a dict for storing binding site
#key: (c,r)
#value:binding site
binding_dict={}        
#----------------------------------
for ind in tqdm(range(len(positive_candidates))):
    gap,pre_visited_yes=3,1           
    detected_center , center_pre=positive_candidates[ind].tolist(),[0,0,0]
    while gap<=15:    
        print(gap)
        #------tovisit stores all subsampled surface points with gap from the center
        tovisit=point_tree.data[point_tree.query_ball_point(detected_center, gap)]
        #------find all sp within this probe is binding site
        visited_yes=[]  
        for point in tovisit:
            #to shorten the computation time   
            if sp_dict[tuple(point)][0]==2:   
                dataset=octforest_rotated(torch.tensor(point),surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,2,device=device,length=16)                          
                [a,b]=evaluation_points3(model,dataset,batch_size=1,device=device)

                sp_dict[tuple(point)]=(a,b)
                if a==1 and b>0.7:
                    visited_yes.append(point)  

            elif sp_dict[tuple(point)][0]==1 and sp_dict[tuple(point)][1]>0.7:
                visited_yes.append(point)              
        if len(visited_yes)==0:
            #delete this candidates
            try:            
                for g in range(3,gap+1):
                    del radio_dict[(tuple(positive_candidates[ind].tolist()),g)]
                    del binding_dict[(tuple(positive_candidates[ind].tolist()),g)]
            except:
                pass
            break
        #-----------------
        #decided if stop the searching earlier        
        if (len(visited_yes)-pre_visited_yes)<5:
            #if stops earlier,delete this candidates
            try:            
                for g in range(3,gap+1):
                    del radio_dict[(tuple(positive_candidates[ind].tolist()),g)]
                    del binding_dict[(tuple(positive_candidates[ind].tolist()),g)]
            except:
                pass
            break                      
        #-------------------------------------------
        #after one loop, determine the ending range 
        detected_bindingATOM,detected_sampled_bindingATOM,detected_center  =sptoatom(visited_yes,protein_atoms_coor) 
        
        if euc_dist(detected_center,center_pre)<0.1:
            ratio=len(visited_yes)/len(tovisit)
            radio_dict[(tuple(positive_candidates[ind].tolist()),gap)]= ratio
            binding_dict[(tuple(positive_candidates[ind].tolist()),gap)]=detected_bindingATOM
            
            gap=gap+1
            pre_visited_yes=len(visited_yes)

        center_pre=detected_center 
delete=[key for key in radio_dict if key[1]<9]
for key in delete:
    del radio_dict[key]
value = 0.5
res_key, res_val = min(radio_dict.items(), key=lambda x: abs(value - x[1]))
inputlist=binding_dict[res_key]
writeOutput(outputname,inputlist)


# In[ ]:




