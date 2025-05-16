"""
# lund_weight.py is a part of the RSA package.
# Copyright (C) 2025 RSA authors (see AUTHORS for details).
# RSA is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
# Added reweighting of sigma parameter

import torch
from torch import nn


class LundWeight(nn.Module):
    def __init__(self, params_base, params, over_sample_factor, device):
        super(LundWeight, self).__init__()
        """
        LundWeight class for computing event weights for the Lund fragmentation function

        Args:
            params_base (torch.Tensor): ------ Base parameters for the Lund fragmentation function
            params (torch.Tensor): ----------- Parameters to be reweighted for the Lund fragmentation function
            over_sample_factor (int): -------- Over-sampling factor for the rejected events
        """

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'

        # Device
        self.device = device

        # Intialize the module parameters
        self.params_base = params_base
        self.params = params

        if self.params['sigma']:
            self.sigma_alt = torch.nn.Parameter(self.params['sigma'].to(self.device), requires_grad = True)

        # Initialize the params dictionary
        # self.params = torch.nn.ParameterDict({})

        # Iterate of all keys in the dictionary and create a parameter for each key
        # fixed_list    = []
        # trainable_list = []
        # for ind, key in enumerate(params_base.keys()):
        #     if key in params:
        #         self.params[key] = torch.nn.Parameter(params[key].to(self.device), requires_grad = True)
        #         # trainable_list.append(ind)
        #     else:
        #         self.params[key] = torch.nn.Parameter(params_base[key].clone().detach().to(self.device), requires_grad = False)
                # fixed_list.append(ind)

        # # fixed_idxs     = torch.tensor(sorted(fixed_list),     dtype=torch.long, device=self.device)
        # # trainable_idxs = torch.tensor(sorted(trainable_list), dtype=torch.long, device=self.device)


        # # emb = list(self.params.keys()) 
        # # emb2idx = {k: i for i, k in enumerate(emb)}
        # # embweight = torch.stack([self.params[k] for k in emb], dim=0)

        # Two lookups (one for 'a', one for 'b')
        self.a_lookup_alt = PIDLookup(params_base, params, prefix='a', device=self.device, alt=True)
        self.b_lookup_alt = PIDLookup(params_base, params, prefix='b', device=self.device, alt=True)
        self.a_lookup_base = PIDLookup(params_base, params, prefix='a', device=self.device, alt=False)
        self.b_lookup_base = PIDLookup(params_base, params, prefix='b', device=self.device, alt=False)



        # Initialize the over-sampling factor
        self.over_sample_factor = over_sample_factor

        # Constants for numerical stability checks
        self.AFROMZERO = 0.02
        self.EXPMAX = 50.
        self.AFROMC = 0.01


    def zMaxCalc(self, a, b, c):
        """
        Estimate the maximum value of z for the Lund fragmentation function
    
        Args:
            a (torch.Tensor): --------------- Parameter a
            b (torch.Tensor): --------------- Parameter b
            c (torch.Tensor): --------------- Parameter c
        
        Returns:
            zMax (torch.Tensor): Estimate for the maximum value of z given (a,b,c)
        """
        # Normalization for Lund fragmentation function so that f <= 1.
        # Special cases for a = 0 and a = c.
        aIsZero = (a < self.AFROMZERO)
        aIsC = (torch.abs(a - c) < self.AFROMC)
        
        # Initialize zMax tensor with the same shape as inputs
        zMax = torch.zeros_like(a, dtype = a.dtype)
        
        # Handle special case where a is zero
        zero_mask = aIsZero
        zMax[zero_mask] = torch.where(c[zero_mask] > b[zero_mask], b[zero_mask] / c[zero_mask], torch.ones_like(zMax[zero_mask], dtype = a.dtype))
        
        # Handle special case where a is essentially equal to c
        c_mask = aIsC & ~zero_mask
        zMax[c_mask] = b[c_mask] / (b[c_mask] + c[c_mask])
        
        # Handle the general case
        general_mask = ~(zero_mask | c_mask)
        if general_mask.any():
            # Compute zMax for the general case
            b_sub = b[general_mask]
            c_sub = c[general_mask]
            a_sub = a[general_mask]
            
            zMax_general = 0.5 * (b_sub + c_sub - torch.sqrt(torch.pow(b_sub - c_sub, 2) + 4 * a_sub * b_sub)) / (c_sub - a_sub)
            
            # Check for NaN values
            if torch.isnan(zMax_general).any():
                print('zMax_1', zMax_general)
                print('a', a[general_mask], 'b', b[general_mask], 'c', c[general_mask])
                exit()
            
            # Adjust zMax in numerically unstable regions 
            zMax_general = torch.where((zMax_general > 0.9999) & (b_sub > 100.), torch.min(zMax_general, 1. - a_sub / b_sub), zMax_general)
            zMax[general_mask] = zMax_general
        
        return zMax

    def likelihood(self, z, mT, a, b, c, z_mask = None, mT_mask = None, a_mask = None, b_mask = None, c_mask = None):
        """
        Compute the likelihood of the Lund fragmentation function

        Args:
            z (torch.Tensor): --------------- Input tensor
            mT (torch.Tensor): -------------- Transverse mass tensor (can be different shape from z)
            a (torch.Tensor): --------------- Parameter a
            b (torch.Tensor): --------------- Parameter b
            c (torch.Tensor): --------------- Parameter c (default: torch.tensor(1., requires_grad=True))
            z_mask (torch.Tensor, optional):  Boolean mask tensor for z (default: None)
            mT_mask (torch.Tensor, optional): Boolean mask tensor for mT (default: None)
            a_mask (torch.Tensor, optional): Boolean mask tensor for a (default: None)
            b_mask (torch.Tensor, optional): Boolean mask tensor for b (default: None)
            c_mask (torch.Tensor, optional): Boolean mask tensor for c (default: None)

        Returns:
            likelihood (torch.Tensor): Computed likelihood values (shape determined by broadcasting rules)
        """
        # Determine the shape after broadcasting
        broadcast_shape = torch.broadcast_shapes(z.shape, mT.shape, a.shape, b.shape, c.shape)
        # If no masks are provided, consider all elements
        if z_mask is None:
            z_mask = torch.ones(z.shape, dtype=torch.bool, device=z.device)
        if mT_mask is None:
            mT_mask = torch.ones(mT.shape, dtype=torch.bool, device=mT.device)
        if a_mask is None:
            a_mask = torch.ones(a.shape, dtype=torch.bool, device=a.device)
        if b_mask is None:
            b_mask = torch.ones(b.shape, dtype=torch.bool, device=b.device)
        if c_mask is None:
            c_mask = torch.ones(c.shape, dtype=torch.bool, device=c.device)

        # Broadcast z, mT, and their masks to the common shape
        z_broad = z.expand(broadcast_shape)
        mT_broad = mT.expand(broadcast_shape)
        a_broad = a.expand(broadcast_shape)
        b_broad = b.expand(broadcast_shape)
        c_broad = c.expand(broadcast_shape)
        z_mask_broad = z_mask.expand(broadcast_shape)
        mT_mask_broad = mT_mask.expand(broadcast_shape)
        a_mask_broad = a_mask.expand(broadcast_shape)
        b_mask_broad = b_mask.expand(broadcast_shape)
        c_mask_broad = c_mask.expand(broadcast_shape)
    
        # Combine masks
        combined_mask = z_mask_broad & mT_mask_broad & a_mask_broad & b_mask_broad & c_mask_broad
        
        # Create a tensor to store the results, initialized with zeros
        likelihood = torch.zeros(broadcast_shape, dtype=z.dtype, device=z.device)
    
        # Only perform calculations on unmasked elements
        z_unmasked = z_broad[combined_mask]
        mT_unmasked = mT_broad[combined_mask]
        a_unmasked = a_broad[combined_mask]
        b_unmasked = b_broad[combined_mask]
        c_unmasked = c_broad[combined_mask]
    
        # Check if we have any unmasked elements to process
        if z_unmasked.numel() > 0:
            # Adjust b-parameter
            b_exp = b_unmasked * mT_unmasked
            
            # Special case for a = 0.
            aIsZero = (a_unmasked < self.AFROMZERO)
            # Determine position of maximum.
            zMax = self.zMaxCalc(a_unmasked, b_exp, c_unmasked)
            
            # Be careful of -inf values in aCoeff, z is being rounded to exactly 1.
            aCoef = torch.log(1. - z_unmasked) - torch.log(1. - zMax)
            if torch.isneginf(aCoef).any():
                print('aCoef is returning -inf value, please check that all z < 1.')
                print('aCoeff', aCoef)
    
            bCoef = (1. / zMax) - (1. / z_unmasked)
            cCoef = torch.log(zMax) - torch.log(z_unmasked)
            fExp = b_exp * bCoef + c_unmasked * cCoef

            
            # Special cases for a = 0.
            not_zero_mask = ~aIsZero
            fExp[not_zero_mask] = fExp[not_zero_mask] + a_unmasked[not_zero_mask] * aCoef[not_zero_mask]
            
            # Feed through numerical stabilizer
            fVal = torch.exp(torch.clamp(fExp, min=-self.EXPMAX, max=self.EXPMAX))
            
            # Assign computed values back to the likelihood tensor
            likelihood[combined_mask] = fVal

        return likelihood

    def sigma_weights(self, px, py, p_mask):
        weights = torch.ones(px.shape, dtype=px.dtype, device=self.device)
        sigma_base = self.params_base['sigma'] / torch.sqrt(torch.tensor(2.0, device=self.device))
        sigma_target = self.sigma_alt / torch.sqrt(torch.tensor(2.0, device=self.device))
        px, py = px[p_mask], py[p_mask]
        kappa = (torch.pow(px,2) + torch.pow(py,2))/(2 * torch.pow(sigma_base, 2))
        ratio = torch.pow(sigma_base,2)/torch.pow(sigma_target,2)
        weights[p_mask] = ratio * torch.exp(-kappa * (ratio-1))

        return weights.prod(axis=1)


    
    def forward(self, z_mT2_pid, fPrel):
        """
        Forward pass of the weight module -- consists of computing the event weights for a given batch
        of training data.

        Args:
            z_mT2 (torch.Tensor): Tensor containing mT2 and z accept-reject data 
            fPrel (torch.Tensor): Tensor containing fPrel values

        Returns:
            weights (torch.Tensor): Computed event weights
        """
        batch_size = z_mT2_pid.shape[0]
        weights = torch.ones(batch_size, device=self.device)


        # z_mT2_pid: Tensor of shape (B, T, 2)
        pid_old = z_mT2_pid[..., 0].abs().round().long()  # → (B, T)
        pid_new = z_mT2_pid[..., 1].abs().round().long()  # → (B, T)

        # Four batched lookups, no Python loops
        a_old_base = self.a_lookup_base(pid_old)
        a_base     = self.a_lookup_base(pid_new)
        b_base     = self.b_lookup_base(pid_new)
        c_base = 1 + a_base - a_old_base

        a_old_alt  = self.a_lookup_alt(pid_old)
        a_alt      = self.a_lookup_alt(pid_new)
        b_alt      = self.b_lookup_alt(pid_new)
        c_alt = 1 + a_alt - a_old_alt


        # # Extract the (absolute value) pid values
        # pid_old = torch.abs(z_mT2_pid[:, :, 0])
        # pid_new = torch.abs(z_mT2_pid[:, :, 1])

        # # Compute and create a, b, and c tensors for base and alternative parameters (This is a bottleneck, there isn't a good way to vectorize)
        # a_old_base = torch.stack([torch.stack([self.params_base[f'a{int(round(pid_old[i,j].item()))}'] for j in range(z_mT2_pid.shape[1])]) for i in range(batch_size)]).view(batch_size, z_mT2_pid.shape[1], 1).to(self.device)
        # a_base = torch.stack([torch.stack([self.params_base[f'a{int(round(pid_new[i,j].item()))}'] for j in range(z_mT2_pid.shape[1])]) for i in range(batch_size)]).view(batch_size, z_mT2_pid.shape[1], 1).to(self.device)
        # b_base = torch.stack([torch.stack([self.params_base[f'b{int(round(pid_new[i,j].item()))}'] for j in range(z_mT2_pid.shape[1])]) for i in range(batch_size)]).view(batch_size, z_mT2_pid.shape[1], 1).to(self.device)
        # c_base = 1 + a_base - a_old_base

        # a_old_alt = torch.stack([torch.stack([self.params[f'a{int(round(pid_old[i,j].item()))}'] for j in range(z_mT2_pid.shape[1])]) for i in range(batch_size)]).view(batch_size, z_mT2_pid.shape[1], 1).to(self.device)
        # a_alt = torch.stack([torch.stack([self.params[f'a{int(round(pid_new[i,j].item()))}'] for j in range(z_mT2_pid.shape[1])]) for i in range(batch_size)]).view(batch_size, z_mT2_pid.shape[1], 1).to(self.device)
        # b_alt = torch.stack([torch.stack([self.params[f'b{int(round(pid_new[i,j].item()))}'] for j in range(z_mT2_pid.shape[1])]) for i in range(batch_size)]).view(batch_size, z_mT2_pid.shape[1], 1).to(self.device)
        # c_alt = 1 + a_alt - a_old_alt
        #check i a_old_alt is on device

        # Create masks for base and alternate parameters (masks should be the same)
        a_mask = a_base != 0.
        b_mask = b_base != 0.
        c_mask = a_base != 0.

        # Extract the mT2 values 
        # mT2 = torch.tensor(z_mT2_pid[:, :, 2], dtype=a_base.dtype)
        mT2 = z_mT2_pid[:, :, 2].clone().detach().to(dtype=a_base.dtype)

        # Reshape into column tensor
        mT2 = mT2.view(mT2.shape[0], mT2.shape[1], 1)
        # Create a mask for zero values
        mT2_mask = mT2 != 0.

        # Extract the accepted z values
        z_accept = z_mT2_pid[:, :, 5]
        # Reshape into column tensor
        z_accept = z_accept.view(z_accept.shape[0], z_accept.shape[1], 1)
        # Remove any zero values
        z_accept_mask = z_accept != 0.

        # Extract the rejected z values
        z_reject = z_mT2_pid[:, :, 6:]
        # Reshape into column tensor
        z_reject = z_reject.view(z_reject.shape[0], z_reject.shape[1], z_reject.shape[2])
        # Remove any zero values along the event index
        z_reject_mask = z_reject != 0.

        # Extract the rejected fPrel values
        fPrel_reject = fPrel[:, :, 1:]
        fPrel_reject = fPrel_reject.view(fPrel_reject.shape[0], fPrel_reject.shape[1], fPrel_reject.shape[2])
        fPrel_reject_mask = fPrel_reject != 0.

        # Compute the accept and reject weights
        accept_weights = self.likelihood(z_accept, mT2, a_alt, b_alt, c_alt, z_mask = z_accept_mask, mT_mask = mT2_mask, a_mask = a_mask, b_mask = b_mask, c_mask = c_mask) \
                         / self.likelihood(z_accept, mT2, a_base, b_base, c_base, z_mask = z_accept_mask, mT_mask = mT2_mask, a_mask = a_mask, b_mask = b_mask, c_mask = c_mask)    
        reject_weights = ((self.over_sample_factor * (fPrel_reject * fPrel_reject_mask.masked_fill(z_accept_mask == 0, 1))) - self.likelihood(z_reject, mT2, a_alt, b_alt, c_alt, z_mask = z_reject_mask, mT_mask = mT2_mask, a_mask = a_mask, b_mask = b_mask, c_mask = c_mask)) \
                         / ((self.over_sample_factor * (fPrel_reject * fPrel_reject_mask.masked_fill(z_accept_mask == 0, 1))) - self.likelihood(z_reject, mT2, a_base, b_base, c_base, z_mask = z_reject_mask, mT_mask = mT2_mask, a_mask = a_mask, b_mask = b_mask, c_mask = c_mask))

        # Flatten the weights
        accept_weights = (accept_weights * z_accept_mask).masked_fill(z_accept_mask == 0, 1).prod(dim=2).prod(dim=1)
        reject_weights = (reject_weights * z_reject_mask).masked_fill(z_reject_mask == 0, 1).prod(dim=2).prod(dim=1)

        # The final event weight is the product of accepted and rejected weights
        weights = accept_weights * reject_weights

        # Add weights for reweighting sigma_pT
        if self.sigma_alt:
            px, py = z_mT2_pid[:, :, 3], z_mT2_pid[:, :, 4]
            p_mask = (px != 0)
            weights_sigma = self.sigma_weights(px,py,p_mask)
            weights = weights * weights_sigma

        return weights


class PIDLookup(nn.Module):
    def __init__(
        self,
        params_base: dict[str, torch.Tensor],
        params:      dict[str, torch.Tensor],
        prefix:      str,               # e.g. 'a' or 'b'
        device:      str = 'cuda',
        alt:         bool = False # True if this is the alternative parameters
    ):
        super().__init__()
        self.device = device

        # For the alternative parameters, the look up table is partly trainable
        if alt:
            # 1) Single pass over both dicts to collect actual PIDs
            all_pids = set()
            # for key in list(params_base.keys()) + list(params.keys()):
            for key in list(params_base.keys()):
                if key.startswith(prefix):
                    pid = int(key[len(prefix):])
                    all_pids.add(pid)

            # 2) Sort and register as a tensor of length V
            self.pid_keys = torch.tensor(
                sorted(all_pids),
                dtype=torch.long,
                device=device
            )                           # shape (V,)
            V = self.pid_keys.size(0)

            # 3) Build the (V, 1) weight matrix, marking frozen rows
            weight = torch.zeros(V, 1, device=device)
            fixed_rows: list[int] = []
            for i, pid in enumerate(self.pid_keys.tolist()):
                key = f'{prefix}{pid}'
                if key in params:
                    # trainable
                    weight[i, 0] = params[key].to(device)
                else:
                    # frozen
                    weight[i, 0] = params_base[key].to(device)
                    fixed_rows.append(i)
            
            # 4) Create an Embedding from this weight (all rows trainable initially)
            self.embed = nn.Embedding.from_pretrained(weight, freeze=False)
            
            # 5) Hook to zero out grads on frozen rows
            fixed_rows_tensor = torch.tensor(fixed_rows, dtype=torch.long, device=device)
            def _freeze_rows(grad: torch.Tensor) -> torch.Tensor:
                grad[fixed_rows_tensor] = 0.
                return grad
            self.embed.weight.register_hook(_freeze_rows)

        # For the base parameters, the look up table is fixed    
        else:
            # 1) Collect all PIDs with the given prefix (e.g., 'a', 'b', etc.)
            all_pids = {
                int(key[len(prefix):])
                for key in params_base.keys()
                if key.startswith(prefix)
            }

            # 2) Sort PIDs and register as tensor of shape (V,)
            self.pid_keys = torch.tensor(
                sorted(all_pids),
                dtype=torch.long,
                device=device
            )
            V = self.pid_keys.size(0)

            # 3) Build the (V, 1) frozen weight matrix
            weight = torch.zeros(V, 1, device=device)
            for i, pid in enumerate(self.pid_keys.tolist()):
                key = f'{prefix}{pid}'
                weight[i, 0] = params_base[key].to(device)

            # 4) Create nn.Embedding with freeze=True (weights won't update)
            self.embed = nn.Embedding.from_pretrained(weight, freeze=True)



    def forward(self, pid_values: torch.LongTensor) -> torch.Tensor:
        """
        pid_values: LongTensor of shape (B, T), arbitrary integers
        Returns:    Tensor of shape (B, T, 1)
        """
        # Map arbitrary PID values → [0..V) with one C/CUDA call
        idxs = torch.searchsorted(self.pid_keys, pid_values)
        # Single fused gather
        return self.embed(idxs)


# class FastPIDModule(nn.Module):
#     def __init__(
#         self,
#         params_base: dict[str, torch.Tensor],
#         params:      dict[str, torch.Tensor],
#         device:      str = 'cuda'
#     ):
#         super().__init__()
#         self.device = device

#         # Two lookups (one for 'a', one for 'b')
#         self.a_lookup = PIDLookup(params_base, params, prefix='a', device=device)
#         self.b_lookup = PIDLookup(params_base, params, prefix='b', device=device)

#     def forward(self, z_mT2_pid: torch.Tensor):
#         # z_mT2_pid: Tensor of shape (B, T, 2)
#         pid_old = z_mT2_pid[..., 0].abs().round().long()  # → (B, T)
#         pid_new = z_mT2_pid[..., 1].abs().round().long()  # → (B, T)

#         # Four batched lookups, no Python loops
#         a_old_base = self.a_lookup(pid_old)
#         a_base     = self.a_lookup(pid_new)
#         b_base     = self.b_lookup(pid_new)

#         a_old_alt  = self.a_lookup(pid_old)
#         a_alt      = self.a_lookup(pid_new)
#         b_alt      = self.b_lookup(pid_new)

#         # Compute c’s
#         c_base = 1 + a_base - a_old_base
#         c_alt  = 1 + a_alt  - a_old_alt

#         return a_old_base, a_base, b_base, c_base, \
#                a_old_alt,  a_alt,  b_alt,  c_alt
