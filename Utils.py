
import torch
import numpy as np
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    print("Set Seed to",seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ref_spectra():
    
    df = pd.read_csv('dfshort.csv')
    df = df.drop(axis=1,columns='Unnamed: 0')
    df = df.iloc[:10,:]
    reference_spectra=np.array(df.values[:,1:],dtype=float)*1000 # scale all species by 1000
    reference_spectra[9,:] = reference_spectra[9,:]/100 # scale water down from 1%
    
    return reference_spectra[:9,:]#*10*np.random.rand(1) # uncomment if training until saturation
    
def Augment_flip(Spectra):

    reference_spectra = get_ref_spectra()
    N = Spectra.shape[0]
    M = reference_spectra.shape[0]
    flipped_spectra = np.flip(reference_spectra,axis=1)
    Con = np.random.random([N,M])
    Con[np.random.random([N,M]) > .5] = 0
    Spectra =  Spectra + torch.tensor(np.matmul(Con,flipped_spectra),device=device, dtype=torch.float)
    
    return Spectra

def Augment_mirror(Spectra):
    
    reference_spectra = get_ref_spectra()
    N = Spectra.shape[0]
    M = reference_spectra.shape[0]
    L = Spectra.shape[1]
    flipped_spectra = np.flip(reference_spectra,axis=1)
    Con = np.random.random([N,M])
    Con[np.random.random([N,M]) > .5] = 0
    mirrored_spectra = np.concatenate((flipped_spectra,reference_spectra), axis=1)
    temp = np.matmul(Con,mirrored_spectra)
    ind = np.random.choice(L,N)
    for i in range(N):
        Spectra[i,:] =  Spectra[i,:] + torch.tensor(temp[i,ind[i]:ind[i]+L],device=device, dtype=torch.float)
    
    return Spectra


def Augment_dilate(Spectra):
    
    reference_spectra = get_ref_spectra()
    N = Spectra.shape[0]
    M = reference_spectra.shape[0]
    L = Spectra.shape[1]
    dilated_spectra = np.flip(reference_spectra,axis=1)
    mirrored_spectra = np.concatenate((dilated_spectra,reference_spectra), axis=1)
    Con = np.random.random([N,M])
    Con[np.random.random([N,M]) > .5] = 0
    ind = np.random.choice(L,M)
    for i in range(M):
        dilated_spectra[i,:] = np.interp(np.arange(ind[i], ind[i]+int(L/2),0.5), np.arange(0, L*2), mirrored_spectra[i,:])
    Spectra =  Spectra + torch.tensor(np.matmul(Con,dilated_spectra),device=device, dtype=torch.float)
    
    return Spectra

def Augment_all(Spectra):
    
    reference_spectra = get_ref_spectra()
    Spectra = Augment_flip(Spectra)
    Spectra = Augment_mirror(Spectra)
    Spectra = Augment_dilate(Spectra)
    
    return Spectra
