
import torch
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
import wandb
import sys


hyperparameter_defaults = dict(
# # Individual
#     lr = 44712591431736535,
#     gamma = 0.1,
#     step_size = 20,
#     seed = 100,
#     N_observations = 1000000,
#     batch_size = 2048,
#     n_epochs = 100,
#     Augment = 'All',
#     interference = True,
#     n_layers = 3,
#     layer_step = 0.30785299519575887,
#     dropout = 0,#0.25076502633258685,
#     kp = 0,
# # All
    lr = 0.000044712591431736535,
#     lr = 0.00044712591431736535,
    gamma = 0.05,
    step_size = 40,
    seed = 100,
    N_observations = 1000000,
    batch_size = 2048,
    n_epochs = 100,
    Augment = 'All',
    interference = True,
    n_layers = 3,
    layer_step = 0.30785299519575887,
    dropout = 0.25076502633258685,
    snr = None,
    TL = False,
# # None
# # class config:
#     lr = 0.0003616702253026447,
#     gamma = 0.1,
#     step_size = 20,
#     seed = 100,
#     N_observations = 1000000,
#     batch_size = 512,
#     n_epochs = 100,
#     Augment = 'None',
#     interference = True,
#     n_layers = 5,
#     n_step = 0.7,
#     dropout = 0.05153182974111975,
#     layer_step=0.4106450990714171,

# # Dilate    
#     lr = 0.0004055962608232558,
#     gamma = 0.1,
#     step_size = 40,
#     seed = 100,
#     N_observations = 1000000,
#     batch_size = 4096,
#     n_epochs = 100,
#     Augment = 'dilate',
#     interference = True,
#     n_layers = 4,
#     layer_step = 0.479310897361274,
#     dropout = 0.060257211302542135,

# # Mirror   
#     lr = 0.00805523198616793,
#     gamma = 0.1,
#     step_size = 20,
#     seed = 100,
#     N_observations = 1000000,
#     batch_size = 256,
#     n_epochs = 100,
#     Augment = 'mirror',
#     interference = True,
#     n_layers = 2,
#     layer_step = 0.571476764300975,
#     dropout = 0.1829413832086243,    
# # flip
#     lr = 0.00026513799133835335,
#     gamma = 0.05,
#     step_size = 80,
#     seed = 100,
#     N_observations = 1000000,
#     batch_size = 4096,
#     n_epochs = 100,
#     Augment = 'flip',
#     interference = True,
#     n_layers = 5,
#     dropout = 0.11927468130000002,
#     layer_step=0.565152635692354,
# # Known
#     lr = 0.0002372506775477351,
#     gamma = 0.1,
#     step_size = 40,
#     seed = 100,
#     N_observations = 1000000,
#     batch_size = 512,
#     n_epochs = 100,
#     Augment = 'known',
#     interference = 'Flase',
#     n_layers = 4,
#     n_step = 0.7,
#     dropout = 0.13727357907536003,
#     layer_step=0.735853178905563,
    
    )

resume = sys.argv[-1] == "--resume"
wandb.init(config=hyperparameter_defaults, project="Interference", entity="emadalibrahim", resume=resume)
config = wandb.config

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from Utils import set_seed
    set_seed(config.seed)

    from data import make_data
    train_loader, valid_loader, test_loader = make_data(config.N_observations,config.seed,config.batch_size,config.snr,config.interference)

    from model import net
    layer = []
    for i in range(config.n_layers-1):
        layer.append(int(1536 * config.layer_step**(i+1)))
    config.hidden = np.array([*layer[:config.n_layers-1],10])
    
    net = net(config).to(device)
    net = nn.DataParallel(net).to(device) 
    
    if config.TL==True:
        net.load_state_dict(torch.load('models/Seed/CNN_seed'+str(config.seed)+'_' +config.Augment+'.pt'))

    wandb.watch(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.step_size,gamma=config.gamma)

    from train_wandb import train
    train(net, train_loader, valid_loader, test_loader, optimizer, scheduler, config)

if __name__ == '__main__':
    main()



