import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

            
            
class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(1287, 256)
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
        
        self.fc1 = nn.Linear(1287, 512)#change this to final dimension 
        self.fc21 = nn.Linear(512, 96)
        self.fc22 = nn.Linear(512, 96)
        self.fc3 = nn.Linear(96, 512)
        self.fc4 = nn.Linear(512, 1287)

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
        self.fc1 = nn.Linear(1287, 256)
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
