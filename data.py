
import torch
import numpy as np
import pandas as pd
import random
from Utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_data(N_observations,seed,batch_size,snr=None,interference = True):
    set_seed(seed)

    df = pd.read_csv('df.csv')
    df = df.drop(axis=1,columns='Unnamed: 0')

    Con = np.random.random([N_observations,df.shape[0]])
    Con[np.random.random([N_observations,df.shape[0]]) > .5] = 0
    ###### Turn off in training ###### 
    if interference == True:
        Con[:int(N_observations*0.8),10:] = 0 # interference not in training set
        
#     Con[:,10]=0
    reference_spectra=np.array(df.values[:,1:],dtype=float)*1000 # scale all species by 1000 (100ppm scaled by 10)
    reference_spectra[9,:] = reference_spectra[9,:]/100 # scale water down (1% scaled by 10)
    dataset=np.matmul(Con,reference_spectra)
    
#     if interference == 'Dirichlet':
#         Con[:,10:]=0
#         df_i = pd.read_csv('df_interfere.csv')
#         reference_spectra_i=np.array(df_i.values[:,1:],dtype=float)*1000 # scale all species by 1000
#         Con_i = np.random.dirichlet(np.ones(df_i.shape[0])/df_i.shape[0], N_observations)*5
#         Con_i = torch.tensor(Con_i)
#         reference_spectra_i = torch.tensor(reference_spectra_i)
#         interference=torch.matmul(Con_i,reference_spectra_i)
#         dataset = np.matmul(Con,reference_spectra) + interference.cpu().detach().numpy()
        
#     if interference == 'PNNL':
#         Con[:,10:]=0
#         df_i = pd.read_csv('df_interfere.csv')
#         reference_spectra_i=np.array(df_i.values[:,1:],dtype=float)*1000 # scale all species by 1000
#         Con_i = np.zeros([N_observations,df_i.shape[0]])
#         for i in range(N_observations):
#             rando = np.random.choice(df_i.shape[0],1)
#             randind = np.random.choice(df_i.shape[0],rando,replace=False)
#             Con_i[i,randind] = np.random.random([rando[0]])
#         Con_i[:int(N_observations*0.8),:] = 0
#         Con_i = torch.tensor(Con_i)
#         reference_spectra_i = torch.tensor(reference_spectra_i)
#         interference=torch.matmul(Con_i,reference_spectra_i)
#         dataset = np.matmul(Con,reference_spectra) + interference.cpu().detach().numpy()
    
    if snr!=None:
        sigma = (10**(-snr/20))*reference_spectra[:10].max(axis=1).min()
        noise = np.random.normal(0,sigma, dataset.shape)
        dataset = dataset + noise

    spectraDF = torch.tensor(dataset)
    conDF = torch.tensor(Con)

    train_features = spectraDF[:int(N_observations*0.6)]#60% of df
    val_features   = spectraDF[int(N_observations*0.6):int(N_observations*0.8)]#20% of df
    test_features  = spectraDF[int(N_observations*0.8):]#20% of df

    train_targets = conDF[:int(N_observations*0.6)]#60% of df
    val_targets   = conDF[int(N_observations*0.6):int(N_observations*0.8)]#20% of df
    test_targets  = conDF[int(N_observations*0.8):]#20% of df

    train = torch.utils.data.TensorDataset(torch.Tensor(np.array(train_features)).to(device), torch.Tensor(np.array(train_targets)).to(device))
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

    val = torch.utils.data.TensorDataset(torch.Tensor(np.array(val_features)).to(device), torch.Tensor(np.array(val_targets)).to(device))
    valid_loader = torch.utils.data.DataLoader(val, batch_size = batch_size)

    test = torch.utils.data.TensorDataset(torch.Tensor(np.array(test_features)).to(device), torch.Tensor(np.array(test_targets)).to(device))
    test_loader = torch.utils.data.DataLoader(test, batch_size =batch_size)
    
    return train_loader, valid_loader, test_loader


def make_data_i(N_observations,seed,batch_size,i,snr):
    set_seed(seed)

    df = pd.read_csv('df.csv')
    df = df.drop(axis=1,columns='Unnamed: 0')

    Con = np.random.random([N_observations,df.shape[0]])
    Con[np.random.random([N_observations,df.shape[0]]) > .5] = 0
    Con[:,10:]=0 # Turn off all interference
    Con[:,10+i] = np.random.random([N_observations]) # Turn on selected species
    reference_spectra=np.array(df.values[:,1:],dtype=float)*1000 # scale all species by 1000
    reference_spectra[9,:] = reference_spectra[9,:]/100 # scale water down from 1%
        
    dataset=np.matmul(Con,reference_spectra)

    if snr!=None:
        sigma = (10**(-snr/20))*reference_spectra[:10].max(axis=1).min()
        noise = np.random.normal(0,sigma, dataset.shape)
        dataset = dataset + noise
    
    test_features = torch.tensor(dataset)
    test_targets = torch.tensor(Con)

    test = torch.utils.data.TensorDataset(torch.Tensor(np.array(test_features)).to(device), torch.Tensor(np.array(test_targets)).to(device))
    test_loader = torch.utils.data.DataLoader(test, batch_size =batch_size)
    
    return test_loader


def make_data_ii(N_observations,seed,batch_size,i,snr):
    set_seed(seed)

    df = pd.read_csv('dfshort.csv')
    df = df.drop(axis=1,columns='Unnamed: 0')

    Con = np.random.random([N_observations,df.shape[0]])
    Con[np.random.random([N_observations,df.shape[0]]) > .5] = 0
    Con[:,10:]=0
#     Con[:,10+i] = np.random.random([N_observations])
    reference_spectra=np.array(df.values[:,1:],dtype=float)*1000 # scale all species by 1000
    reference_spectra[9,:] = reference_spectra[9,:]/100 # scale water down from 1%
    
    dataset=np.matmul(Con,reference_spectra)
    
    df_i = pd.read_csv('df_interfere.csv')
    reference_spectra_i=np.array(df_i.values[:,1:],dtype=float)*1000 # scale all species by 1000
    Con_i = np.zeros([N_observations,df_i.shape[0]])
    Con_i[:,i] = np.random.random([N_observations])
    dataset = dataset + np.matmul(Con_i,reference_spectra_i)
    
    if snr!=None:
        sigma = (10**(-snr/20))*reference_spectra[:10].max(axis=1).min()
        noise = np.random.normal(0,sigma, dataset.shape)
        dataset = dataset + noise
    
    test_features = torch.tensor(dataset)
    test_targets = torch.tensor(Con)

    test = torch.utils.data.TensorDataset(torch.Tensor(np.array(test_features)).to(device), torch.Tensor(np.array(test_targets)).to(device))
    test_loader = torch.utils.data.DataLoader(test, batch_size =batch_size)
    
    return test_loader
