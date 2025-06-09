#surfaceSAS
from VoxelProt.surfaceGeneration.curvatures import *
import torch
import torch.nn.functional as F
import numpy as np
import pykeops
from pykeops.torch import LazyTensor
from VoxelProt.surfaceGeneration.dictionary import distVDW
def cal_atomtype_radius(protein_atoms):
    '''
          assign each atom with the Van der Waals radius(pm)
          [C,H,O,N,S,Se, P,CL,IOD,Na,K,MG,ZN,CA,FE,others]
          [0,1,2,3,4,5,  6,7,  8, 9,10,11,12,13,14,15]
          [170,120,152,155,180,190,180,175,198,227,275,173,139,231,244,100]
          Args:
              protein_atoms(list): (N,3) atom type.

          Returns:
              Tensor: (N,1) :atom type
    ''' 
      
    atom_type=[0 for each in protein_atoms]
    for i in range(len(atom_type)):
        s3=str(protein_atoms[i])[6:][0:3]
        s2=str(protein_atoms[i])[6:][0:2]
        s1=str(protein_atoms[i])[6:][0:1]
  
        if s3=="IOD":
          atom_type[i]=8     
        elif s2=="NA":
          atom_type[i]=9  
        elif s2=="MG":
          atom_type[i]=11 
        elif s2=="ZN":
          atom_type[i]=12  
        elif s2=="CA":
          atom_type[i]=13  
        elif s2=="FE":
          atom_type[i]=14   
        elif s2=="CL":
          atom_type[i]=7 
        elif s2=="SE":
          atom_type[i]=5
        elif s1=="K":
          atom_type[i]=10    
        elif s1=="H":
          atom_type[i]=1
        elif s1=="C":
          atom_type[i]=0
        elif s1=="N":
          atom_type[i]=3
        elif s1=="O":
          atom_type[i]=2
        elif s1=="S":
          atom_type[i]=4
        elif s1=="P":
          atom_type[i]=6
        else:
          atom_type[i]=15

    return atom_type 

def sample(atoms, distance=0.5,sup_sampling=40):
    """generate a point cloud {X1,…Xnb}, where n is the number of atoms and b =20
    draw n*b points at random in the neighborhood of atoms
    Args:
      atoms (Tensor): (n,3) coordinates of the atoms.
      distance (float, optional): value of the level set to sample from
        the smooth distance function. Defaults to 1.05.
      sup_sampling=40
    
    return:the coordinates of all atoms
    """
    if atoms.size()[0]==0:
        return None
    num,dim=atoms.shape

    # sample n*b points at random in the neighborhood of our atoms
    sample_points = atoms[:, None, :] + 10 * distance * torch.randn(num, sup_sampling, dim).type_as(atoms)
    sample_points = sample_points.view(-1, dim)  # (N*B, D)
    return sample_points

def soft_distances(x,y,smoothness=0.01):
    """Computes a soft distance function to the atom centers of a protein.
      
    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        smoothness (float, optional): atom radii.Defaults to .01.
    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    # (N, 1, 3) atoms
    x_i = LazyTensor(x[:, None, :])
    # (1, M, 3) sampling points
    y_j = LazyTensor(y[None, :, :])
    # (N, M, 1) squared distances
    D_ij = ((x_i - y_j) ** 2).sum(-1)  

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    #return -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(-1)
    return -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(-1)

def normalVector(points, atoms, smoothness=0.01):
    """Computes a normal vector for each surface point, which points out always.
    
    Args:
        points: (N,3) surfacePoints.
        atoms: (M,3) protein atoms.
    Returns:
        Tensor: (N,3) normal vector for each surface point 
    """   
    p = points.detach()
    p.requires_grad = True
    dists = soft_distances(atoms,p,smoothness=smoothness)
    Loss = (1.0 * dists).sum()
    g = torch.autograd.grad(Loss, p)[0]
    return F.normalize(g, p=2, dim=-1) 

def gradDescent(protein_atoms_coor,samples,distance,smoothness=0.01,n_iter=4):
    with torch.enable_grad():
        if samples.is_leaf:
            samples.requires_grad = True 
        #Iterative loop: gradient descent along the potential with respect to the positions samples
        for it in range(n_iter):
            dists = soft_distances( protein_atoms_coor, samples,smoothness)            
            Loss = ((dists - distance) ** 2).sum()
            g = torch.autograd.grad(Loss, samples)[0]
            samples.data -= 0.5 * g 
    return samples

def removingOutliers(protein_atoms_coor,samples,distance,n_iter=4,smoothness=0.01,variance=0.001):
    #1)Only keep the points which are reasonably close to the level set:
    dists = soft_distances(protein_atoms_coor,samples,smoothness=smoothness)
    margin = (dists-distance).abs()
    mask = margin < variance * distance
    #2) Remove insides points.
    zz = samples.detach()
    zz.requires_grad = True
    for it in range(n_iter):
        dists = soft_distances(protein_atoms_coor,zz,smoothness=smoothness)

        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss,  zz)[0]

        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
        zz =  zz + 1.0 * distance * normals    
    dists = soft_distances(protein_atoms_coor,zz,smoothness=smoothness)
    mask = mask & (dists > 1.5 * distance)
    samples = samples[mask].contiguous().detach() 
    return samples


def createSurface(atomscoor,atomtypes_protein,batch=None,sup_sampling=5,smoothness=0.01,variance=0.001,n_iter=4, probe=1.4,scales=[1.5],threshold=-0.7,reg=0.01,device="cuda"):
    """generate the SAS surface point model and assign each point with 
    a normal vector.
    The normal vector points toward the convex direction
    """
    # store the coor of each type of atoms in protein_atoms_coor0...15
    for each in range(16):
        vars()["atomscoor"+str(each)] =np.empty((0,3))
    '''
    #old one, sometimes got errors
    for each in range(len(atomtypes_protein)):
        vars()["atomscoor"+str(atomtypes_protein[each])] = \
                np.append(vars()["atomscoor"+str(atomtypes_protein[each])],\
                           np.array([atomscoor[each]]), axis=0)
    '''
    for each in range(len(atomtypes_protein)):
        vars()["atomscoor"+str(atomtypes_protein[each])] =                 np.append(vars()["atomscoor"+str(atomtypes_protein[each])],                           np.array(atomscoor[each].reshape(1,3)), axis=0)
        
    for each in range(16):
        vars()["protein_atoms_coor"+str(each)] = torch.from_numpy(vars()["atomscoor"+str(each)]).to(device=device)
        vars()["protein_atoms_coor"+str(each)] =vars()["protein_atoms_coor"+str(each)].detach().contiguous()

    #Concatenates all samples together  
    samples=torch.from_numpy(np.empty((0,3))).to(device=device)
    
    # store the coor of samples of each type of atoms in samples0...15
    for each in range(16):
        vars()["samples"+str(each)] =sample(vars()["protein_atoms_coor"+str(each)])        
      
    ##generating the surface of each type of atoms with outliers  
        if vars()["samples"+str(each)]==None:
            continue
        vars()["samples"+str(each)]=vars()["samples"+str(each)].detach().contiguous()
        vars()["samples"+str(each)]=        gradDescent(vars()["protein_atoms_coor"+str(each)],                    vars()["samples"+str(each)],distVDW[each]-1,smoothness=smoothness,n_iter=n_iter)

        ##Removing the outliers as well as the points inside the proteins
        vars()["samples"+str(each)]=        removingOutliers(vars()["protein_atoms_coor"+str(each)],                         vars()["samples"+str(each)],distVDW[each]-1,n_iter=n_iter,                         smoothness=smoothness,variance=variance)

        
        samples= torch.cat((samples,vars()["samples"+str(each)]))  
    #rolls a 'water' probe ball (1.4 Å diameter) over the Van der Waals surface
    atoms_SAS=samples.to(device="cuda").to(torch.float32)
    atoms_SAS=atoms_SAS.detach().contiguous()
    # sampling points around each surface points
    
    samples_SAS=sample(atoms_SAS,sup_sampling=sup_sampling)    

    samples_SAS =samples_SAS.detach().contiguous()

    #generate the SAS with outliers 
    samples_SAS=gradDescent(atoms_SAS,samples_SAS,probe,smoothness=smoothness,n_iter=n_iter)
   
    ##Removing the outliers as well as the points inside the proteins
    remainPoints=removingOutliers(atoms_SAS,samples_SAS,probe,n_iter=n_iter,smoothness=smoothness,variance=variance)
    #find the normal vector for each surface point
    nor_vector=normalVector(remainPoints,atoms_SAS,smoothness=smoothness)
    shape_index=curvatures(remainPoints, scales=scales, batch=batch, normals=nor_vector, reg=reg)
    shape_index3=curvatures(remainPoints, scales=[3.0], batch=batch, normals=nor_vector, reg=reg)  
    
    candidate_index=(shape_index<threshold).nonzero().squeeze()
    # calcualte the oreented_nor_vector
      
    a=torch.mul(nor_vector[:,0], shape_index)
    b=torch.mul(nor_vector[:,1], shape_index)
    c=torch.mul(nor_vector[:,2], shape_index)
    oriented_nor_vector=torch.cat((a.view(shape_index.shape[0],1), b.view(shape_index.shape[0],1), c.view(shape_index.shape[0],1)), 1)

    return remainPoints,candidate_index,shape_index,oriented_nor_vector,shape_index3

