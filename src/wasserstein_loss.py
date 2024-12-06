"""
# wasserstein_loss.py is a part of the RSA package.
# Copyright (C) 2024 RSA authors (see AUTHORS for details).
# RSA is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import torch
import ot

class WassersteinLoss(torch.nn.Module):
    def __init__(self, p):
        super(WassersteinLoss, self).__init__()
        """
        Compute the one-dimensional Wasserstein distance between two input tensors x and y with weights x_weights and y_weights

        Args:
            p (int): Order of the Wasserstein distance

        Returns:
            (torch.tensor): One-dimensional Wasserstein distance
        """
        self.p = p

    def forward(self, x, y, x_weights = None, y_weights = None):
        """
        Compute the one-dimensional Wasserstein distance between two input tensors x and y
        with x weighted by x_weights
        """
        # The weights must be normalized for the Wasserstein loss
        x_weights = x_weights / torch.sum(x_weights)
        y_weights = torch.ones(y.shape[0]) / y.shape[0]
        # Compute the Wasserstein distance
        return ot.wasserstein_1d(x, y, u_weights = x_weights, v_weights = y_weights, p = self.p)