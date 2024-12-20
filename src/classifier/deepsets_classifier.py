"""
# deepsets_classifier.py is a part of the RSA package.
# Copyright (C) 2024 RSA authors (see AUTHORS for details).
# RSA is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class DeepSetsClassifier(nn.Module):
    def __init__(self, input_dim, phi_hidden_dim=64, rho_hidden_dim=64,
                 phi_layers=3, rho_layers=3, device=torch.device("cuda"),
                 dropout_prob=0.2, mask_pad=False,  momentum=0.1):
        super(DeepSetsClassifier, self).__init__()
        """
        Container class for the DeepSets classifier.

        Args:
            input_dim (int): Dimension of the input data
            phi_hidden_dim (int): Dimension of the hidden layers in the phi network
            rho_hidden_dim (int): Dimension of the hidden layers in the rho network
            phi_layers (int): Number of layers in the phi network
            rho_layers (int): Number of layers in the rho network
            device (torch.device): Device to run the model on
            dropout_prob (float): Dropout probability for the networks
            mask_pad (bool): Apply phi to padding or ignore
            momentum (float): Momentum for batch normalization
        """
        
        self.mask_pad = mask_pad
        s = 0.1 # PyTorch default 0.01 works well for multiplicity
        
        # Define phi network (element-wise processing)
        phi_layers_list = [nn.Linear(input_dim, phi_hidden_dim),
                           nn.BatchNorm1d(phi_hidden_dim, momentum=0.05, eps=1e-6),
                           nn.LeakyReLU(negative_slope=s),
                           nn.LayerNorm(phi_hidden_dim),
                           nn.Dropout(dropout_prob)]
        
        for _ in range(1, phi_layers-1):
            phi_layers_list.extend([
                nn.Linear(phi_hidden_dim, phi_hidden_dim),
                nn.BatchNorm1d(phi_hidden_dim, momentum=0.05, eps=1e-6), 
                nn.LeakyReLU(negative_slope=s),
                nn.LayerNorm(phi_hidden_dim),
                nn.Dropout(dropout_prob)
            ])
        phi_layers_list.extend([
                nn.Linear(phi_hidden_dim, phi_hidden_dim),
                nn.BatchNorm1d(phi_hidden_dim, momentum=0.05, eps=1e-6), 
                nn.LeakyReLU(negative_slope=s),
                nn.LayerNorm(phi_hidden_dim)])
        
        self.phi = nn.Sequential(*phi_layers_list).to(device)

        # Define rho network (permutation-invariant aggregation)
        rho_layers_list = []
        for _ in range(rho_layers - 1):
            rho_layers_list.extend([
                nn.Linear(phi_hidden_dim, rho_hidden_dim),
                nn.BatchNorm1d(rho_hidden_dim, momentum=0.05, eps=1e-6),
                nn.LeakyReLU(negative_slope=s),
                nn.LayerNorm(rho_hidden_dim),
                nn.Dropout(dropout_prob)
            ])
            phi_hidden_dim = rho_hidden_dim
        
        # Append Linear layer
        rho_layers_list.append(nn.Linear(rho_hidden_dim, 1))

        self.rho = nn.Sequential(*rho_layers_list).to(device)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Number of learnable parameters: {total_params}')
                    
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_loss = []
        self.val_loss = []
        
        self.device = device
    
    def forward(self, x, mask=None, device=None):
        """
        Forward function for the DeepSets classifier.

        Args:
            x (torch.Tensor): Input data
            mask (torch.Tensor): Mask for the input data
            device (torch.device): Device to run the model on

        Returns:
            output (torch.Tensor): Output of the DeepSets classifier (score)
        """
        if device == None: 
            device = self.device
        else:
            print(device)
            self.rho = self.rho.to(device) 
            self.phi = self.phi.to(device)
            
        x = x.to(device)
        batch_size, n_particles, input_dim = x.size()
        
        # Create mask for non-zero entries if not provided
        if mask is None: mask = (x != 0).any(dim=-1)  # shape: (batch_size, n_particles)
        else: mask = mask.to(x.device)
        # Separate non-zero entries and padding
        non_zero_mask = mask
        # Process non-zero entries
        non_zero_x = x[non_zero_mask]  # shape: (num_non_zero, input_dim)
        processed_non_zero_x = self.phi(non_zero_x)  # shape: (num_non_zero, output_dim)
        
        # Initialize output tensor
        output = torch.zeros(batch_size, n_particles, processed_non_zero_x.size(-1), device=x.device)
        output[non_zero_mask] = processed_non_zero_x
            
        if self.mask_pad:
            # If you want phi to apply to the padded entries
            # Aggregate the results (summing along the particle dimension)
            output = output.sum(dim=1)
        else:
            output = output.mean(dim=1)

        # Apply rho to the aggregated set
        output = self.rho(output)
        
        # Move tensors back to CPU and detach
        x.cpu().detach()
        mask.cpu().detach()

        # Optional: Free GPU memory explicitly
        del x, mask

        return output.squeeze()
    
    def train_classifier(self, train_exp_loader, train_sim_loader, val_exp_loader, val_sim_loader, 
                         device, num_epochs=10, learning_rate=0.001, pretraining_epochs=0):
        """
        Training loop for the DeepSets classifier.

        Args:
            train_exp_loader (DataLoader): DataLoader for the experimental data
            train_sim_loader (DataLoader): DataLoader for the simulated data
            val_exp_loader (DataLoader): DataLoader for the experimental data (validation)
            val_sim_loader (DataLoader): DataLoader for the simulated data (validation)
            device (torch.device): Device to run the model on
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for the optimizer
            pretraining_epochs (int): Number of pretraining epochs with MSE loss
        """
        # Choose optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Initialize epoch counter
        epoch_n = 0

        # Training loop
        for epoch in range(num_epochs + pretraining_epochs):
            
            self.train().to(device)
            train_loss = 0.0
            train_exp_loss = 0.0
            train_sim_loss = 0.0
            
            train_exp_loader_tdqm = tqdm(train_exp_loader, desc="Training", leave=False)
            # Training phase
            # optimizer.zero_grad()
            for (exp_batch, sim_batch) in zip(train_exp_loader_tdqm, train_sim_loader):
                
                optimizer.zero_grad()
                
                exp_data = exp_batch[0]#.to(device)
                exp_mask = exp_batch[1]#.to(device)
                sim_data = sim_batch[0]#.to(device)
                sim_mask = sim_batch[1]#.to(device)

                # Combine the data and masks
                combined_data = torch.cat([exp_batch[0], sim_batch[0]], dim=0)
                combined_mask = torch.cat([exp_batch[1], sim_batch[1]], dim=0)

                # Forward pass with the combined data
                combined_output = self(combined_data, combined_mask)

                # Separate the outputs, should do more gracefully in preprocessing
                exp_output = combined_output[:exp_batch[0].size(0)] 
                sim_output = combined_output[exp_batch[0].size(0):]
                
                # Labeling and loss calculation:
                if epoch_n < pretraining_epochs:
                    # Gaussian of mean 0.5 and standard deviation of 0.5
                    exp_output = torch.sigmoid(exp_output)
                    sim_ouput = torch.sigmoid(sim_output)
                    std = 0.2
                    exp_labels = torch.randn_like(exp_output) * std + 0.5
                    sim_labels = torch.randn_like(sim_output) * std + 0.5

                    mse_loss = nn.MSELoss()
                    loss_exp = mse_loss(exp_output, exp_labels)
                    loss_sim = mse_loss(sim_output, sim_labels)
                else:
                    exp_labels = torch.ones(exp_output.size(0), dtype=exp_output.dtype, device=exp_output.device)
                    sim_labels = torch.zeros(sim_output.size(0), dtype=sim_output.dtype, device=sim_output.device)

                    loss_exp = self.criterion(exp_output, exp_labels)
                    loss_sim = self.criterion(sim_output, sim_labels)
                loss = (loss_exp + loss_sim) / 2

                # Backpropagate and update weights
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * exp_data.size(0)
                train_exp_loss += loss_exp.item() * exp_data.size(0)
                train_sim_loss += loss_sim.item() * sim_data.size(0)
                
            # Validation analytics
            self.eval()
            val_loss = 0.0
            val_exp_loss = 0.0
            val_sim_loss = 0.0
            all_labels = []
            all_predictions = []
            train_sim_loader_tdqm = val_sim_loader#tqdm(val_sim_loader, desc="Validating", leave=False)
            with torch.no_grad():
                for (exp_batch, sim_batch) in zip(train_sim_loader_tdqm, val_sim_loader):
                
                    exp_data = exp_batch[0]
                    exp_mask = exp_batch[1]
                    sim_data = sim_batch[0]
                    sim_mask = sim_batch[1]

                    exp_output = self(exp_data, exp_mask)
                    sim_output = self(sim_data, sim_mask)
                    
                    # Labeling and loss calculation:
                    if epoch_n < pretraining_epochs:
                    # Gaussian of mean 0.5 and standard deviation of 0.5
                        exp_output = torch.sigmoid(exp_output)
                        sim_ouput = torch.sigmoid(sim_output)
                        std = 0.2
                        exp_labels = torch.randn_like(exp_output) * std + 0.5
                        sim_labels = torch.randn_like(sim_output) * std + 0.5
                        
                        mse_loss = nn.MSELoss()
                        loss_exp = mse_loss(exp_output, exp_labels)
                        loss_sim = mse_loss(sim_output, sim_labels)
                    else:
                        exp_labels = torch.ones(exp_output.size(0), dtype=exp_output.dtype, device=exp_output.device)
                        sim_labels = torch.zeros(sim_output.size(0), dtype=sim_output.dtype, device=sim_output.device)

                        loss_exp = self.criterion(exp_output, exp_labels)
                        loss_sim = self.criterion(sim_output, sim_labels)

                        # Collect labels and predictions for ROC AUC calculation
                        all_labels.extend(exp_labels.cpu().numpy())
                        all_labels.extend(sim_labels.cpu().numpy())
                        all_predictions.extend(torch.sigmoid(exp_output).cpu().numpy())
                        all_predictions.extend(torch.sigmoid(sim_output).cpu().numpy())

                    loss = (loss_exp + loss_sim) / 2

                    val_loss += loss.item() * exp_data.size(0)
                    val_exp_loss += loss_exp.item() * exp_data.size(0)
                    val_sim_loss += loss_sim.item() * sim_data.size(0)

            # Calculate ROC AUC score
            roc_auc = roc_auc_score(all_labels, all_predictions)

            train_loss /= len(train_exp_loader.dataset)
            train_exp_loss /= len(train_exp_loader.dataset)
            train_sim_loss /= len(train_sim_loader.dataset)
            val_loss /= len(val_exp_loader.dataset)
            val_exp_loss /= len(val_exp_loader.dataset)
            val_sim_loss /= len(val_sim_loader.dataset)

            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

            if epoch_n < pretraining_epochs:
                print(f'Pretraining epoch [{epoch+1}/{num_epochs}]')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.5g}, Train Exp Loss: {train_exp_loss:.4g}, Train Sim Loss: {train_sim_loss:.4g}')
            print(f'Val Loss: {val_loss:.5g}, Val Exp Loss: {val_exp_loss:.4g}, Val Sim Loss: {val_sim_loss:.4g}')
            print(f'ROC AUC Score: {roc_auc:.8f}')
            epoch_n += 1

            
    def evaluate_model(self, test_exp_loader, test_sim_loader):
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for exp_batch, sim_batch in zip(test_exp_loader, test_sim_loader):
                exp_data, = exp_batch
                sim_data, = sim_batch

                exp_output = classifier(exp_data)
                sim_output = classifier(sim_data)

                exp_labels = torch.ones(exp_output.size(0))
                sim_labels = torch.zeros(sim_output.size(0))

                loss_exp = self.criterion(exp_output, exp_labels)
                loss_sim = self.criterion(sim_output, sim_labels)
                loss = (loss_exp + loss_sim) / 2

                test_loss += loss.item() * exp_data.size(0)

        print(f'Test Loss: {test_loss/len(test_exp_loader.dataset):.4f}')
        
    def loss_plot(self):

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label='Training Loss', marker='o')
        plt.plot(self.val_loss, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

#--------------------------------------------------------------------#
#----------------- Auxiliary data preparation functions -------------#
#--------------------------------------------------------------------#

def prepare_data(exp_obs, sim_obs, batch_size=10000, num_workers=4, pin_memory=True):
    # Calculate masks for non-zero entries
    exp_mask = (exp_obs != 0).any(dim=-1)
    sim_mask = (sim_obs != 0).any(dim=-1)
    
    # Create datasets with observations and masks
    exp_dataset = TensorDataset(
        exp_obs.clone().detach().requires_grad_(False),
        exp_mask.clone().detach()
    )
    sim_dataset = TensorDataset(
        sim_obs.clone().detach().requires_grad_(False),
        sim_mask.clone().detach()
    )
    # Create data loaders
    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return exp_loader, sim_loader
    
def min_max_scaling(outputs, new_min=-5, new_max=5):
    # Calculate the min and max values of the outputs
    min_val = torch.min(outputs)
    max_val = torch.max(outputs)
    
    # Apply the min-max scaling formula
    scaled_outputs = new_min + (outputs - min_val) / (max_val - min_val) * (new_max - new_min)
    
    return scaled_outputs

#--------------------------------------------------------------------#
#----------------- Auxiliary plotting functions ---------------------#
#--------------------------------------------------------------------#

def plot_score_histogram(exp_scores, sim_scores, sim_weights=None, same_bins=False, bins=50):
    """
    Plot a histogram of classifier scores for the given data.
    
    Parameters:
    - exp_scores: numpy array or torch.Tensor of scores from the experimental data
    - sim_scores: numpy array or torch.Tensor of scores from the simulated data
    - sim_weights: numpy array or torch.Tensor of weights for the simulated scores, or None
    - same_bins: bool, whether to use the same bins for both histograms
    - bins: int, number of bins for the histogram
    """
    exp_scores = np.array(exp_scores)
    sim_scores = np.array(sim_scores)
    
    if sim_weights is not None:
        sim_weights = np.array(sim_weights)
    
    if same_bins:
        # Compute the bin edges from the combined range
        min_score = min(exp_scores.min(), sim_scores.min())
        max_score = max(exp_scores.max(), sim_scores.max())
        bin_edges = np.linspace(min_score, max_score, bins + 1)
    else:
        bin_edges = bins
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for experimental scores as a line
    plt.hist(exp_scores, bins=bin_edges, density=True, histtype='step', linewidth=2, label="Truth")

    # Plot histogram for simulated scores with weights as a line
    if sim_weights is not None:
        plt.hist(sim_scores, bins=bin_edges, density=True, weights=sim_weights, histtype='step', linewidth=2, label="Base (Reweighted)")
    else:
        plt.hist(sim_scores, bins=bin_edges, density=True, histtype='step', linewidth=2, label="Base")
    
    plt.title('Histogram of Classifier Scores (On Test Set)')
    plt.xlabel('Truth score')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def mult_classifier_plot(exp_scores, sim_scores):

    unique_exp_scores, exp_counts = np.unique(exp_scores, return_counts=True)
    unique_sim_scores, sim_counts = np.unique(sim_scores, return_counts=True)

    exp_bar_width = np.median(np.diff(unique_exp_scores))
    sim_bar_width = np.median(np.diff(unique_sim_scores))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(unique_exp_scores, exp_counts / np.sum(exp_counts), width=exp_bar_width, alpha=0.7, label='Truth')
    ax.bar(unique_sim_scores, sim_counts / np.sum(sim_counts), width=sim_bar_width, alpha=0.7, label='Base')

    ax.set_xlabel('Scores')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.show()