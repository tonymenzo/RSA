"""
# wasserstein_loss.py is a part of the RSA package.
# Copyright (C) 2024 RSA authors (see AUTHORS for details).
# RSA is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import torch
import ot
from ot.sliced import sliced_wasserstein_distance

class WassersteinLoss_nD(torch.nn.Module):
    def __init__(self, device, p=2, n_projections=50, seed=42):
        super(WassersteinLoss_nD, self).__init__()
        """
        Compute the n-dimensional sliced Wasserstein distance between two input tensors X and Y with weights x_weights and y_weights

        Args:
            p (int): Order of the sliced Wasserstein distance

        Returns:
            (torch.tensor): One-dimensional Wasserstein distance
        """
        self.p = p
        self.n_projections = n_projections
        self.seed = seed

        # Device
        self.device = device

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'

    def forward(self, X, Y, x_weights = None, y_weights = None):
        """
        Compute the sliced Wasserstein distance between two input tensors X and Y
        with X weighted by x_weights
        """
        # Transpose data for sliced_wasserstein function
        X, Y = X.T, Y.T
        # The weights must be normalized for the Wasserstein loss
        x_weights = x_weights / torch.sum(x_weights)
        y_weights = torch.ones(Y.shape[0], device=self.device) / Y.shape[0]
        # Compute the Wasserstein distance
        return sliced_wasserstein_distance(X, Y, a=x_weights, b=y_weights, n_projections=self.n_projections, p=self.p)