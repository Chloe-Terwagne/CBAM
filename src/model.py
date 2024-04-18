import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, data: dict):
        """
        :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
                                                'logits', 'wt_logits', 'wt_embedding']
        :return: predicted DMS score
        """
        x = data['embedding']
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.sum(x, dim=1)
        x = self.fc2(x)
        return x
    
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(1280, 512)#change this to final dimension 
        self.fc21 = nn.Linear(512, 96)
        self.fc22 = nn.Linear(512, 96)
        self.fc3 = nn.Linear(96, 512)
        self.fc4 = nn.Linear(512, 1280)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, data: dict):
        x = data['embedding']
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, data: dict):
        """
        :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
                                                'logits', 'wt_logits', 'wt_embedding']
        :return: predicted DMS score
        """
        x = data['embedding']
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.sum(x, dim=1)
        x = self.fc2(x)
        return x
