import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os,random,csv,sys
from VoxelProt.trainModel.vgg import VGG_net 
from VoxelProt.trainModel.pickleoctree import *


"""proteins=["1a27"]
bindingoctree_add="/home/llab/Downloads/nonBS_octree_N/"
nonbindingoctree_add="/home/llab/Downloads/BS_octree_N/"
FILE ="Voxelprot"
device="cuda"
epoches=1
"""
proteins=[]
with open(sys.argv[1], mode ='r')as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        proteins.append(lines[0])

bindingoctree_add=sys.argv[2]
nonbindingoctree_add=sys.argv[3]
FILE =sys.argv[4]
epoches=int(sys.argv[5])
device="cuda"


train_set=[]
for each_file in os.listdir(bindingoctree_add):
    if(each_file[0:4] in proteins):
        train_set.append({"octree":bindingoctree_add+each_file,"label":1})

for each_file in os.listdir(nonbindingoctree_add):
    if(each_file[0:4] in proteins):
        train_set.append({"octree":nonbindingoctree_add+each_file,"label":0})
random.shuffle(train_set) 


lr=0.001
#config_model
model=VGG_net(channel_in=16,num_classes=2).to(device)

#config_optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epoches):  # loop over the dataset multiple times

    for data in range(len(train_set)):
            inputs=readOctree(train_set[data]["octree"])  

            labels=torch.tensor([train_set[data]["label"]]).to(device)

                #zero the parameter gradients
            optimizer.zero_grad()

                #forward 
            outputs = model(inputs,batch_size=labels.shape[0])

            pred = torch.argmax(outputs, dim=1)
                #calculate the loss
            log_softmax = F.log_softmax(outputs,dim=1)

            loss = F.nll_loss(log_softmax, labels)
            #backward
            loss.backward()

                #optimize
            optimizer.step()           

    torch.save(model.state_dict(),FILE+str(epoch)+".pth")   

