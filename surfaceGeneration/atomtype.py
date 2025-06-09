
import numpy as np
from sklearn.neighbors import KDTree
import torch

def scale_atom(protein_atoms, protein_atoms_res):
    '''
         assign each atom with a value:
         H:0, C_aliphatic:1, C_aromatic:2, N:3, O:4, P:5, S:6, Se:7, halogen(CL,IOD):8 and metal(Na,K,MG or ZN or MN or CA or FE):9
         # in TRP: CD1,CD2,CG,CE2,CZ2,CZ2,CH2,CZ3,CE3
         # in PHE: CZ,CE1,CD1,CG,CD2,CE2
         # in TYR: CZ,CE2,CD2,CD1,CE1,CG
         Args:
             protein_atoms(list): (N,3) atom type.
             protein_atoms_res (list): (N,1) the residue name of each atom.
          
         Returns:
             Tensor: (N,1) :atom type
    ''' 
    trp=["CD1","CD2","CG","CE2","CZ2","CZ2","CH2","CZ3","CE3"]
    phe=["CZ","CE1","CD1","CG","CD2","CE2"]
    tyr=["CZ","CE2","CD2","CD1","CE1","CG"]
    metal=["NA","MG","ZN","MN","CA","FE"]
    rv=[0 for each in protein_atoms]
    for i in range(len(rv)):

        s3=str(protein_atoms[i])[6:][0:3]
        s2=str(protein_atoms[i])[6:][0:2]
        s1=str(protein_atoms[i])[6:][0:1]
        #C_aromatic:2
        if protein_atoms_res[i]=="TRP" and ((s3 in trp) or (s2 in trp)):
            rv[i]=2
        elif protein_atoms_res[i]=="PHE" and ((s3 in phe) or (s2 in phe)):
            rv[i]=2      
        elif protein_atoms_res[i]=="TYR" and ((s3 in tyr) or (s2 in tyr)):
            rv[i]=2   
        #metal(NA,K,MG or ZN or MN or CA or FE):9
        elif (s1=="K") or (s2 in metal):
            rv[i]=9
        #halogen(CL,IOD):8
        elif (s2=="CL") or (s3=="IOD"):
            rv[i]=8
        #H:0, C_aliphatic:1, N:3, O:4, P:5, S:6, SE:7,
        elif s1=="H":
            rv[i]=0
        elif s1=="C":
            rv[i]=1
        elif s1=="N":
            rv[i]=3
        elif s1=="O":
            rv[i]=4
        elif s1=="S":
            rv[i]=6
        elif s1=="P":
            rv[i]=5
        elif s2=="SE":
            rv[i]=7
    return rv

def getAtomType(protein_atoms, protein_atoms_res,protein_atoms_coor,surfacePoints):
    '''get the atom type for each atom
        Args:
            protein_atoms(list): (N,3) atom type.
            protein_atoms_res (list): (N,1) the residue name of each atom.
            surfacePoints(Tensor): (M,3) surface point coors.
        
        Returns:
            Tensor: (M,1) :the atomtype for each surface point
    '''
    atom_type=scale_atom(protein_atoms, protein_atoms_res)
    atom_features=torch.zeros(surfacePoints.size()[0],10)

    kdt = KDTree(protein_atoms_coor.detach().numpy(), metric='euclidean')
    nearest=kdt.query(surfacePoints,1,return_distance=False)[:,0]
    for ind in range(len(surfacePoints)):
        atom_features[ind][atom_type[nearest[ind]]]=1
    return atom_features

