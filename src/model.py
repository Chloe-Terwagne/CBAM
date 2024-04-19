import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
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

            
# class ProteinModel(nn.Module):
#     def __init__(self):
#         super(ProteinModel, self).__init__()
#         self.fc1 = nn.Linear(1281, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 64)
#         self.fc4 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, data: dict):
#         """
#         :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
#                                                 'logits', 'wt_logits', 'wt_embedding']
#         :return: predicted DMS score
#         """
#         x = data['embedding']
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = torch.sum(x, dim=1)
#         x = self.fc4(x)
#         return x

    
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


# class EmbeddingCNN(nn.Module):
#     def __init__(self):
#         super(EmbeddingCNN, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

#         # Fully connected layers
#         self.fc1 = nn.Linear(256, 256)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(256, 1)

#     def forward(self, data: dict):
#         """
#         :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
#                                                 'logits', 'wt_logits', 'wt_embedding']
#         :return: predicted DMS score
#         """
#         x = data['embedding'].unsqueeze(1)  # Add a channel dimension
#         # Dynamically calculate the sequence length for each input tensor
#         sequence_lengths = torch.tensor([emb.shape[1] for emb in data['embedding']])
        
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.relu(x)

#         # Global average pooling across the sequence dimension
#         x = nn.functional.adaptive_avg_pool1d(x, 1).squeeze(2)

#         # Fully connected layers
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1287, out_channels=580, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=580, out_channels=92, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=92, out_channels=580, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=580, out_channels=1287, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(1287, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        # Learning rate scheduler
        # self.scheduler = ReduceLROnPlateau(Adam(self.parameters()), mode='min', factor=0.1, patience=5, verbose=True)

    def forward(self, data: dict):
        """
        :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
                                                'logits', 'wt_logits', 'wt_embedding']
        :return: predicted DMS score
        """
        x = data['embedding']  # Convert to PyTorch tensor

        x = x.permute(0, 2, 1)  # Rearrange dimensions to [batch_size, sequence_length, 1281]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        # Global average pooling across the sequence dimension
        x = nn.functional.adaptive_avg_pool1d(x, 1).squeeze(2)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def update_lr_scheduler(self, val_loss):
        self.scheduler.step(val_loss)
