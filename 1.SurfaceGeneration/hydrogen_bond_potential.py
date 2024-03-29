# -*- coding: utf-8 -*-
"""hydrogen_bond_potential.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15AA7GjRO1GRJ9--cdyh5tCiSAYZ8UGF3
"""

"""
Hydrogen bond potential
These values range from −1 (optimal position for a hydrogen bond acceptor) 
to +1 (optimal position for a hydrogen bond donor).
"""
import numpy as np
from math import pi
from sklearn.neighbors import KDTree
import torch
import Bio
from Bio import SeqIO, SearchIO, Entrez
from Bio.PDB import *

##dictionary to contain all info
# This  polar hydrogen's names correspond to that of the program Reduce. 
polarHydrogens = {}
polarHydrogens["ALA"] = ["H"]
polarHydrogens["GLY"] = ["H"]
polarHydrogens["SER"] = ["H", "HG"]
polarHydrogens["THR"] = ["H", "HG1"]
polarHydrogens["LEU"] = ["H"]
polarHydrogens["ILE"] = ["H"]
polarHydrogens["VAL"] = ["H"]
polarHydrogens["ASN"] = ["H", "HD21", "HD22"]
polarHydrogens["GLN"] = ["H", "HE21", "HE22"]
polarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
polarHydrogens["HIS"] = ["H", "HD1", "HE2"]
polarHydrogens["TRP"] = ["H", "HE1"]
polarHydrogens["PHE"] = ["H"]
polarHydrogens["TYR"] = ["H", "HH"]
polarHydrogens["GLU"] = ["H"]
polarHydrogens["ASP"] = ["H"]
polarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
polarHydrogens["PRO"] = []
polarHydrogens["CYS"] = ["H"]
polarHydrogens["MET"] = ["H"]
polarHydrogens["PCA"] = []





# Dictionary from an H atom to its donor atom.
donorAtom = {}
donorAtom["H"] = "N"
# ARG
# ARG NHX
# Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
# ARG NE
# Angle: ~ 120 NE, HE, point, 180 degrees
donorAtom["HH11"] = "NH1"
donorAtom["HH12"] = "NH1"
donorAtom["HH21"] = "NH2"
donorAtom["HH22"] = "NH2"
donorAtom["HE"] = "NE"
 # ASN
 # Angle ND2,HD2X: 180
 # Plane: CG,ND2,OD1
 # Angle CG-OD1-X: 120
donorAtom["HD21"] = "ND2"
donorAtom["HD22"] = "ND2"
  # GLU
  # PLANE: CD-OE1-OE2
  # ANGLE: CD-OEX: 120
  # GLN
  # PLANE: CD-OE1-NE2
  # Angle NE2,HE2X: 180
  # ANGLE: CD-OE1: 120
donorAtom["HE21"] = "NE2"
donorAtom["HE22"] = "NE2"

  # HIS Donors: ND1, NE2
  # Angle ND1-HD1 : 180
  # Angle NE2-HE2 : 180
donorAtom["HD1"] = "ND1"
donorAtom["HE2"] = "NE2"

  # TRP Donor: NE1-HE1
  # Angle NE1-HE1 : 180
donorAtom["HE1"] = "NE1"

  # LYS Donor NZ-HZX
  # Angle NZ-HZX : 180
donorAtom["HZ1"] = "NZ"
donorAtom["HZ2"] = "NZ"
donorAtom["HZ3"] = "NZ"

  # TYR donor: OH-HH
  # Angle: OH-HH 180
donorAtom["HH"] = "OH"
  # SER donor:
  # Angle: OG-HG-X: 180
donorAtom["HG"] = "OG"

  # THR donor:
  # Angle: OG1-HG1-X: 180
donorAtom["HG1"] = "OG1"









# Dictionary from acceptor atom to a third atom on which to compute the plane.
acceptorPlaneAtom = {}
acceptorPlaneAtom["O"] = "CA"
  # ASN Acceptor
acceptorPlaneAtom["OD1"] = "CB"
  # ASP
  # Plane: CB-CG-OD1
  # Angle CG-ODX-point: 120
acceptorPlaneAtom["OD2"] = "CB"
  # GLU
  # PLANE: CD-OE1-OE2
  # ANGLE: CD-OEX: 120
  # GLN
  # PLANE: CD-OE1-NE2
  # Angle NE2,HE2X: 180
  # ANGLE: CD-OE1: 120
acceptorPlaneAtom["OE1"] = "CG"
acceptorPlaneAtom["OE2"] = "CG"
  # HIS Acceptors: ND1, NE2
  # Plane ND1-CE1-NE2
  # Angle: ND1-CE1 : 125.5
  # Angle: NE2-CE1 : 125.5
acceptorPlaneAtom["ND1"] = "NE2"
acceptorPlaneAtom["NE2"] = "ND1"
  # TYR acceptor OH
  # Plane: CE1-CZ-OH
  # Angle: CZ-OH 120
acceptorPlaneAtom["OH"] = "CE1"

  # TYR donor: OH-HH
  # Angle: OH-HH 180
acceptorPlaneAtom["OH"] = "CE1"



# Dictionary from an acceptor atom to its directly bonded atom on which to
# compute the angle.
acceptorAngleAtom = {}
acceptorAngleAtom["O"] = "C"
acceptorAngleAtom["O1"] = "C"
acceptorAngleAtom["O2"] = "C"
acceptorAngleAtom["OXT"] = "C"
acceptorAngleAtom["OT1"] = "C"
acceptorAngleAtom["OT2"] = "C"
  # ASN Acceptor
acceptorAngleAtom["OD1"] = "CG"
  # Angle CG-ODX-point: 120
acceptorAngleAtom["OD2"] = "CG"

  # GLU
  # PLANE: CD-OE1-OE2
  # ANGLE: CD-OEX: 120
  # GLN
  # PLANE: CD-OE1-NE2
  # Angle NE2,HE2X: 180
  # ANGLE: CD-OE1: 120

acceptorAngleAtom["OE1"] = "CD"
acceptorAngleAtom["OE2"] = "CD"

  # HIS Acceptors: ND1, NE2
  # Plane ND1-CE1-NE2
  # Angle: ND1-CE1 : 125.5
  # Angle: NE2-CE1 : 125.5
acceptorAngleAtom["ND1"] = "CE1"
acceptorAngleAtom["NE2"] = "CE1"
  # TYR acceptor OH
  # Plane: CE1-CZ-OH
  # Angle: CZ-OH 120
acceptorAngleAtom["OH"] = "CZ"
  # SER acceptor:
  # Angle CB-OG-X: 120
acceptorAngleAtom["OG"] = "CB"
  # THR acceptor:
  # Angle: CB-OG1-X: 120
acceptorAngleAtom["OG1"] = "CB"



def computeAngleDeviation(a, b, c, theta):
  """Compute the absolute value of the deviation from theta
    Args:
        a, b and c: the corr of points
        theta: the ideal theta are based on data
        
    Returns:
        angle(pi)
  """
  return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)



def computeSatisfied_CO_HN(atoms):
  """Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
    Args:
        atoms(list):(N,3) the coor of each protein atoms
        
    Returns:
        satisfied_CO(set):
        satisfied_HN(set):
  """

  ns = NeighborSearch(atoms)
  satisfied_CO = set()
  satisfied_HN = set()

  for atom1 in atoms:
    res1 = atom1.get_parent()
    if atom1.get_id() == "O":
      neigh_atoms = ns.search(atom1.get_coord(), 3, level="A")
      for atom2 in neigh_atoms:
        if atom2.get_id() == "H":
          res2 = atom2.get_parent()
          # Ensure they belong to different residues.
          if res2.get_id() != res1.get_id():
            # Compute the angle N-H:O, ideal value is 160 
            angle_N_H_O_dev = computeAngleDeviation(res2["N"].get_coord(),atom2.get_coord(),atom1.get_coord(),16 * np.pi / 18)
            # Compute angle H:O=C, ideal value is 150
            angle_H_O_C_dev = computeAngleDeviation(atom2.get_coord(),atom1.get_coord(),res1["C"].get_coord(),15 * np.pi / 18)
            ## Allowed deviations:
            if ( angle_N_H_O_dev<= 3*np.pi / 18 and angle_H_O_C_dev <= 5.5* np.pi / 18):
              satisfied_CO.add(res1.get_id())
              satisfied_HN.add(res2.get_id())

  return satisfied_CO, satisfied_HN


def isPolarHydrogen(atom_name, res_type):
  """
    to determine if a hydrogen atom in a residue is polar hydrogen or not
    Args:
        atom_name: hydrogen atom's name, "HZ1"
        res_type: residue name,like "VAL"
        
    Returns:
        True or False 
  """
  if res_type in polarHydrogens:
    if atom_name in polarHydrogens[res_type]:
       return True
  else:
    return False

def isAcceptorAtom(atom_name, res):
  """
    to determine if a atom in a residue is acceptor or not, like 'O' and nitrogen in "HIS"
    Args:
        atom_name: hydrogen atom's name, "HZ1"
        res: a Residue object         
    Returns:
        True or False 
  """
  if atom_name.startswith("O"):
    return True
  else:
    if res.get_resname() == "HIS":
      if atom_name == "ND1" and "HD1" not in res:
        return True
      if atom_name == "NE2" and "HE2" not in res:
        return True
  return False


def computeAnglePenalty(angle_deviation,std_dev): 
  """
    to determine angle_deviation from ideal value.
    Args:
        angle_deviation: angel deviation
        std_dev: default to pi         
    Returns:
        a penalty value to each point to show its hydrogen potential 
  """
  return max(0.0, 1.0 - (angle_deviation / std_dev)**2 )



 
def computeChargeHelper(atom_name, res, v):
  """
    Compute the charge of a surface point v.
    Args:
        atom_name: hydrogen atom's name, "HZ1"
        res: a Residue object 
        v(numpy): the coor of a surface point       
    Returns:
        Charge 
  """

  res_type = res.get_resname()
  # check if it is a polar hydrogen
  if isPolarHydrogen(atom_name, res_type):
    donor_atom = donorAtom[atom_name]
    a = res[donor_atom].get_coord()  # N/O
    b = res[atom_name].get_coord()  # H
    # Donor-H is always 180.0 degrees, = pi
    angle_deviation = computeAngleDeviation(a, b, v, np.pi)
    angle_penalty = computeAnglePenalty(angle_deviation,18*np.pi / 18)
    return 1.0 * angle_penalty

  elif isAcceptorAtom(atom_name, res):
    acceptor_atom = res[atom_name]
    b = acceptor_atom.get_coord()
    try:
      a = res[acceptorAngleAtom[atom_name]].get_coord()
    except:
      return 0.0
    # 120 degress for acceptor
    angle_deviation = computeAngleDeviation(a, b, v, 2 * np.pi / 3)
    angle_penalty = computeAnglePenalty(angle_deviation,18*np.pi / 18)
    return -1.0 * angle_penalty
  return 0.0



def computeCharge(surfacePoints, protein_atoms_coor, protein_atoms):
  """
    Compute the charge of all surface points.
    Args:
        surfacePoints(Tensor): (M,3) surface point coors. 
        protein_atoms_coor (Tensor): (N,3) atom coors.
        protein_atoms: a list of Atom objects   
    Returns:
        Charge(tensor) (N,1): charge for each surface point 
  """
  satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(protein_atoms)
  
  #get all residues info without non-standard 
  residues = {}
  reses=[item.get_parent() for item in protein_atoms]
  for res in reses:
    chain_id = res.get_parent().get_id()
    if chain_id == "":
      chain_id = " "
    residues[(chain_id, res.get_id())] = res
  #using charge to store charge values
  charge = np.array([0.0] * (surfacePoints.size()[0]))
  
  kdt = KDTree(protein_atoms_coor[:,0:3].detach().numpy(), metric='euclidean')
  nearest=kdt.query(surfacePoints,1,return_distance=False)

  for i in range(np.shape(nearest)[0]):
    atom_name=protein_atoms[nearest[i].item()].get_id()
    res_id=protein_atoms[nearest[i].item()].get_parent().get_id() 
    chain_id= protein_atoms[nearest[i].item()].get_full_id()[2]
    if chain_id == "":
      chain_id = " "
    if atom_name == "H" and res_id in satisfied_HN:
      continue 
    if atom_name == "O" and res_id in satisfied_CO:
      continue 
    charge[i] = computeChargeHelper(atom_name, residues[(chain_id, res_id)], surfacePoints[i].numpy())

  '''
  ## red - donor
  ## del starting: after showing
  import matplotlib.pyplot as plt
  import random
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure(figsize=(14, 12))
  ax = fig.add_subplot(projection='3d')
  x=surfacePoints[:,0]
  y=surfacePoints[:,1]
  z=surfacePoints[:,2]
  t=charge
  p=ax.scatter(x,y,z,c=t)
  plt.set_cmap('bwr')
  fig.colorbar(p,ax=ax)
  plt.show()


  ## del ending

  '''
  return (torch.tensor(charge)).view(-1,1)
