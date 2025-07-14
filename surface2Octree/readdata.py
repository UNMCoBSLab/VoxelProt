from csv import reader
import csv
import torch
def readSurfacePts(addSurface):
    data=[]
    with open(addSurface, "r") as csvfile:
        spamreader = reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    surfacePoints=torch.Tensor(data)[:,0:3]
    return surfacePoints
    
def readData(addSurface,addKD,addElec,addCandidates,addAtom):
    """Args:
    address: e.g. "/home/llab/Desktop/JBLab/task1/features/surfaceNormals/"+train_ADP_name[0]+".csv"
    num: the num of selected points
    return:surfacePoints,oriented_nor_vector,KD_Hpotential,elec,candidates
    """
    data=[]
    with open(addSurface, "r") as csvfile:
        spamreader = reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    surfacePoints=torch.Tensor(data)[:,0:3]
    oriented_nor_vector=torch.Tensor(data)[:,3:6]

    data=[]
    with open(addKD, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    KD_Hpotential=torch.Tensor(data)  


    data=[]
    with open(addElec,newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    elec=torch.Tensor(data)  
        
        
    data=[]
    with open(addCandidates, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    candidatesInd=torch.Tensor(data)
    candidates=[surfacePoints[int(each[0])] for each in candidatesInd]


    data=[]
    with open(addAtom,newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    atomtype=torch.Tensor(data)  
    
    return surfacePoints,oriented_nor_vector,KD_Hpotential,elec,atomtype,candidates

def readSI(addSI):
    data=[]
    with open(addSI, "r") as csvfile:
        spamreader = reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data.append([float(each) for each in row[0].split(",")])
    return torch.Tensor(data)

