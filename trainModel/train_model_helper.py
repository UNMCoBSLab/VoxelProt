import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os,random,csv,sys
from VoxelProt.trainModel.ResNet import ResNet 
from VoxelProt.trainModel.SE_ResNet import SE_ResNet 
from VoxelProt.trainModel.vgg import VGG_net 
from VoxelProt.trainModel.pickleoctree import *
def get_train_set_chen11(bindingoctree_add,nonbindingoctree_add):
    train_set=[]
    for each_file in tqdm(os.listdir(bindingoctree_add)):
        train_set.append({"octree":os.path.join(bindingoctree_add, each_file),"label":1})
    
    for each_file in tqdm(os.listdir(nonbindingoctree_add)):
        train_set.append({"octree":os.path.join(nonbindingoctree_add, each_file),"label":0})
    random.shuffle(train_set)  
    return train_set
    
def get_validation_list_joined():
    rv = []
    fn = os.path.join(os.getcwd(), "VoxelProt", "dataset", "joined-prt2lig.csv")
    with open(fn, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: rv.append(row[0])
    return rv
    
def get_list(n_fold,train_test_val):
    p_list= os.path.join(os.getcwd(), "VoxelProt", "dataset", "cross_val_splits", f"fold_{n_fold}", f"{train_test_val}.txt")
    with open(p_list, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return [each[0:4] for each in ids]

def get_train_set(bindingoctree_add,nonbindingoctree_add,proteins_train):
    train_set=[]
    for each_file in tqdm(os.listdir(bindingoctree_add)):
        if(each_file[0:4] in proteins_train):        
            train_set.append({"octree":os.path.join(bindingoctree_add,each_file),"label":1})
    
    for each_file in tqdm(os.listdir(nonbindingoctree_add)):
        if(each_file[0:4] in proteins_train):
            train_set.append({"octree":os.path.join(nonbindingoctree_add,each_file),"label":0})
    random.shuffle(train_set)  
    return train_set
        
def evaluation_point(model,dataset_cg,label,dataset_v = None, device="cuda"):
    with torch.no_grad():
        yes=0
        no=0

        for ind in range(len(dataset_cg)):
                inputs_cg=readOctree(dataset_cg[ind])
                labels=torch.tensor([label]).to(device) 
                if dataset_v !=None:
                    inputs_v=readOctree(dataset_v[ind])
                    outputs = model(inputs_cg,inputs_v,batch_size=labels.shape[0])
                else:
                    outputs = model(inputs_cg,batch_size=labels.shape[0])
                pred = torch.argmax(outputs, dim=1)
                if (pred==0).item():
                    no=no+1
                else:
                    yes=yes+1

        if yes>=no:
            pred=torch.tensor([1]).to(device)
        else:
            pred=torch.tensor([0]).to(device) 

        return pred.eq(torch.tensor([label]).to(device)).float().mean()    
        

def train_one_epoch(train_set,model,optimizer,device):
    for data in tqdm(range(len(train_set))):
        try:          
            inputs=readOctree(train_set[data]["octree"])  
            labels=torch.tensor([train_set[data]["label"]]).to(device)
            #zero the parameter gradients
            optimizer.zero_grad()
            #forward 
            outputs = model(inputs,batch_size=labels.shape[0])
            log_softmax = F.log_softmax(outputs,dim=1)
            loss = F.nll_loss(log_softmax, labels)
            #backward
            loss.backward()
            optimizer.step()           
        except:
            print(f"errors {train_set[data]} ")  
    return model,optimizer
    
def eval(protein_list, model, bindingoctree_add, nonbindingoctree_add, candidate_number= 11): 
    ac_negative, num1= 0, 0
    for each in tqdm(protein_list):
        for iii in range(candidate_number):
            for ii in range(10):
                try: 
                    #dataset_cg=[nonbindingoctree_add+each +str(iii)+str(ii)+str(i)+".pkl" for i in range(10)]
                    dataset_cg=[os.path.join(nonbindingoctree_add, f"{each}{iii}{ii}{i}.pkl") for i in range(10)]
                    ac_negative = evaluation_point(model,dataset_cg,0)+ac_negative
                    num1=num1+1
                except:
                    continue   
    print("negative_acc_point_validation1: "+str(ac_negative/num1))
    
    #=============================================================== 
   
    ac_positive, num = 0,0 
    for each in tqdm(protein_list):
        for ii in range(1500):
            try: 
                #dataset_cg=[bindingoctree_add+each +str(ii)+str(i)+".pkl" for i in range(10)]
                dataset_cg=[os.path.join(bindingoctree_add, f"{each}{ii}{i}.pkl") for i in range(10)]
                ac_positive=evaluation_point(model,dataset_cg,1)+ac_positive
                num=num+1
            except:
                continue                
    print("postive_acc_point_validation: "+str(ac_positive/num))     
    
def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    
    #Restore model weights
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    # Restore optimizer state
    start_epoch = 0
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    return model, optimizer, start_epoch
    
    
def train_model(model_type,epoches,n_fold,bindingoctree_add,nonbindingoctree_add,lr=0.001,candidate_number= 11,checkpoint_path = None, device="cuda"):
    proteins_train = get_list(n_fold,"train")
    proteins_val = get_list(n_fold,"val")
    train_set = get_train_set(bindingoctree_add,nonbindingoctree_add,proteins_train)  
    
    #config_model
    if model_type =="ResNet":
        model=ResNet(channel_in=16,num_classes=2).to(device)        
    elif model_type == "vgg":
        model=VGG_net(channel_in=16,num_classes=2).to(device)
    elif model_type == "SE_ResNet":
        model=SE_ResNet(channel_in=16,num_classes=2).to(device)
    #config_optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #load model 
    if checkpoint_path!=None:
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)
    #train
    for epoch in range(start_epoch, epoches):  
        print(f"epoch: {epoch}")
        model,optimizer = train_one_epoch(train_set,model,optimizer,device)
        checkpoint = {
            'epoch':           epoch,                  
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{model_type}_{epoch}_{n_fold}.pth")
        eval(proteins_val, model, bindingoctree_add, nonbindingoctree_add, candidate_number= 11)  
        
def train_model_chen11(model_type,epoches,chen11_octree_add,joined_octree_add,lr=0.001,checkpoint_path = None,device="cuda"):
    chen11_bindingoctree_add = os.path.join(chen11_octree_add, "BS_octree_cg")
    joined_bindingoctree_add = os.path.join(joined_octree_add, "BS_octree_cg")
    chen11_nonbindingoctree_add = os.path.join(chen11_octree_add, "nonBS_octree_cg")
    joined_nonbindingoctree_add = os.path.join(joined_octree_add, "nonBS_octree_cg")
    
    proteins_val = get_validation_list_joined()
    train_set = get_train_set_chen11(chen11_bindingoctree_add,chen11_nonbindingoctree_add)  
    
    #config_model
    if model_type =="ResNet":
        model=ResNet(channel_in=16,num_classes=2).to(device)        
    elif model_type == "vgg":
        model=VGG_net(channel_in=16,num_classes=2).to(device)
    elif model_type == "SE_ResNet":
        model=SE_ResNet(channel_in=16,num_classes=2).to(device)
    #config_optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #load model 
    start_epoch = 0
    if checkpoint_path!=None:
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)
    #train
    for epoch in range(start_epoch, epoches):  
        print(f"epoch: {epoch}")
        model,optimizer = train_one_epoch(train_set,model,optimizer,device)
        checkpoint = {
            'epoch':           epoch,                  
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{model_type}_{epoch}_chen11.pth")
        eval(proteins_val, model, joined_bindingoctree_add, joined_nonbindingoctree_add, candidate_number= 5)  
