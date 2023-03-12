import numpy as np
import wandb
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import Augment_all, Augment_flip, Augment_mirror, Augment_dilate
from data import make_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, train_loader, valid_loader, test_loader, optimizer, scheduler, config):
    
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    losses = {'train':[], 'validation':[], 'test':[]}

    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, config.n_epochs+1):
#         train_loader, valid_loader, test_loader = make_data(config.N_observations,config.seed,config.batch_size,config.interference)
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0

        ###################
        # train the model #
        ###################
        net.train()
        for Spectra, Con in train_loader:
            Spectra=Spectra.to(device)
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            if config.Augment=='All':
                Spectra = Augment_all(Spectra)
            if config.Augment=='flip':
                Spectra = Augment_flip(Spectra)
            if config.Augment=='mirror':
                Spectra = Augment_mirror(Spectra)
            if config.Augment=='dilate':
                Spectra = Augment_dilate(Spectra)
            out = net(Spectra)
            loss = criterion(out,Con[:,:10])
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()    
            # update training loss
            train_loss += loss.item()*Spectra.size(0)


        ######################    
        # validate the model #
        ######################
        net.eval()
        for Spectra, Con in valid_loader:
            Spectra=Spectra.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            out = net(Spectra)
            # calculate the batch loss
            loss = criterion(out,Con[:,:10])
            # update average validation loss 
            valid_loss += loss.item()*Spectra.size(0)

        ######################    
        # test the model #
        ######################
        net.eval()
        for Spectra, Con in test_loader:
            Spectra=Spectra.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            out = net(Spectra)
            # calculate the batch loss
            loss =  criterion2(out,Con[:,:10])
            # update average validation loss 
            test_loss += loss.item()*Spectra.size(0)


        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        test_loss = test_loss/len(test_loader.dataset)
        scheduler.step()

        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss) 
        losses['test'].append(test_loss)    
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}, \tTest Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss, test_loss))
        
        metrics = {'training/loss': train_loss,'validation/loss': valid_loss,'test/loss': test_loss}

        wandb.log(metrics)
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))

            torch.save(net.state_dict(), 'models/AT5_CNN_seed' + str(config.seed) + '_SNR' + str(config.snr) + '_' + config.Augment + '.pt')
            valid_loss_min = valid_loss
        
