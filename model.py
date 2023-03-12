import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self,config):
        super(net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size = 8, stride = 8, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size = 4, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size = 4, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.Linear = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1536,config.hidden[0]),
            nn.ReLU(),
        )
        modules = []
        for i in range(len(config.hidden)-1):
            modules.append(nn.Dropout(p=config.dropout))
            modules.append(nn.Linear(config.hidden[i], config.hidden[i+1]))
#             if i != (len(config.hidden)-2): # Comment out at inference
            modules.append(nn.ReLU())
        self.body = nn.Sequential(*modules)
    
    def forward(self, s):
        s = s[:,:,None].permute([0,2,1])
        s = self.net(s)
        flat = s.flatten(start_dim=1)
        out = self.Linear(flat)
        out = self.body(out)
        return out
