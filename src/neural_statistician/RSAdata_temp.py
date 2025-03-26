import numpy as np
import torch
import os

from torch.utils import data
from sklearn.model_selection import train_test_split



# class SyntheticSetsDataset(data.Dataset):
#     def __init__(self, n_datasets, sample_size, n_features, distributions):
#         self.n_datasets = n_datasets
#         self.sample_size = sample_size
#         self.n_features = n_features
#         self.distributions = distributions

#         self.data = self.create_sets()
#         self.datasets = self.data['datasets']

#     def __getitem__(self, item):
#         return self.datasets[item]

#     def __len__(self):
#         return self.n_datasets

#     def generate_distribution(self, distribution):
#         m = np.random.uniform(-1, 1)
#         v = np.random.uniform(0.5, 2)

#         if distribution == 'gaussian':
#             samples = np.random.normal(m, v, (self.sample_size, self.n_features))
#             return samples, m, v

#         elif distribution == 'mixture of gaussians':
#             mix_1 = np.random.normal(-(1 + np.abs(m)), v / 2, (int(self.sample_size / 2),
#                                                                self.n_features))
#             mix_2 = np.random.normal((1 + np.abs(m)), v / 2, (int(self.sample_size / 2),
#                                                               self.n_features))
#             return np.vstack((mix_1, mix_2)), 1 + np.abs(m), v / 2

#         elif distribution == 'exponential':
#             samples = np.random.exponential(1, (self.sample_size, self.n_features))
#             return self.augment_distribution(samples, m, v), m, v

#         elif distribution == 'reverse exponential':
#             samples = - np.random.exponential(1, (self.sample_size, self.n_features))
#             return self.augment_distribution(samples, m, v), m, v

#         elif distribution == 'laplacian':
#             samples = np.random.laplace(m, v, (self.sample_size, self.n_features))
#             return samples, m, v

#         elif distribution == 'uniform':
#             samples = np.random.uniform(-1, 1, (self.sample_size, self.n_features))
#             return self.augment_distribution(samples, m, v), m, v

#         elif distribution == 'negative binomial':
#             samples = np.random.negative_binomial(50, 0.5, (self.sample_size,
#                                                             self.n_features))
#             samples = np.asarray(samples, dtype=np.float64)
#             return self.augment_distribution(samples, m, v), m, v

#         else:
#             print("Unrecognised choice of distribution.")
#             return None

#     @staticmethod
#     def augment_distribution(samples, m, v):
#         aug_samples = samples.copy()
#         aug_samples -= np.mean(samples)
#         aug_samples /= np.std(samples)
#         aug_samples *= v ** 0.5
#         aug_samples += m
#         return aug_samples

#     def create_sets(self):
#         sets = np.zeros((self.n_datasets, self.sample_size, self.n_features),
#                         dtype=np.float32)
#         labels = []
#         means = []
#         variances = []

#         for i in range(self.n_datasets):
#             distribution = np.random.choice(self.distributions)

#             x, m, v = self.generate_distribution(distribution)

#             sets[i, :, :] = x
#             labels.append(distribution)
#             means.append(m)
#             variances.append(v)

#         return {
#             "datasets": sets,
#             "labels": np.array(labels),
#             "means": np.array(means),
#             "variances": np.array(variances),
#             "distributions": np.array(self.distributions)
#         }


class RSA_Dataset(data.Dataset):
    """
	Converts observable dataset into PyTorch syntax.


    data_sample: 1 event
    data_sample_size: Size of 1 event -> # of Hadrons + Padding
    n_datasets: # of generated events
    n_features: # of low-level features
	"""


    def __init__(self, data_paths, sample_size, n_features, test_size, random_state=42, shuffle= True):
        # self.n_datasets = n_datasets        # Number of dataset
        self.sample_size = sample_size      # Size of max multiplicity
        self.n_features = n_features        # Number of low-level observables used
        # self.distributions = distributions
        self.data_paths = data_paths
        self.data_paths_len = len(data_paths)

        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

        self.data = self.create_sets()
        self.datasets = self.data['datasets']

    def __getitem__(self, item):
        return self.datasets[item]

    def __len__(self):
        return self.n_datasets
    
    # def prescale(exp_data, sim_data, axes=(0, 1)):
    #     """
    #     Prescale the experimental and simulated data using the combined mean and standard deviation.

    #     Args:
    #         exp_data (np.ndarray): The experimental data.
    #         sim_data (np.ndarray): The simulated data.
    #         axes (tuple): The axes along which to calculate the mean and standard deviation.

    #     Returns:
    #         np.ndarrays: The prescaled experimental and simulated data.
    #     """
    #     # Mask to identify non-padded entries (i.e., entries that are not [0.0, 0.0, 0.0, 0.0])
    #     non_padded_mask_exp = ~(np.all(exp_data == 0, axis=-1))
    #     non_padded_mask_sim = ~(np.all(sim_data == 0, axis=-1))
        
    #     # Flatten the non-padded parts of the datasets along the specified axes for mean/std calculation
    #     combined_data = np.concatenate([exp_data[non_padded_mask_exp], sim_data[non_padded_mask_sim]], axis=0)
    #     combined_mean = combined_data.mean(axis=0)
    #     # print("Mean:", combined_mean)
    #     combined_std = combined_data.std(axis=0)

    #     # Scale only the non-padded entries using the combined mean and std
    #     exp_data_scaled = np.copy(exp_data)
    #     sim_data_scaled = np.copy(sim_data)
    #     exp_data_scaled[non_padded_mask_exp] = (exp_data[non_padded_mask_exp] - combined_mean) / combined_std
    #     sim_data_scaled[non_padded_mask_sim] = (sim_data[non_padded_mask_sim] - combined_mean) / combined_std
        
    #     return exp_data_scaled, sim_data_scaled
    
    def prepare_datasets(self, data_paths):
        
        data = []
        labels = []
        for d_path in self.data_paths:
            hadrons = np.load(d_path, mmap_mode="r")
            filename = os.path.basename(d_path)

            # multiplicity = np.array([len(hadrons[i,:][np.abs(hadrons[i,:,0]) > 0.0]) for i in range(len(hadrons))])
            # hadrons = hadrons[:, :multiplicity, :]

            # Extract (px, py, pz)
            px, py, pz = hadrons[..., 0], hadrons[..., 1], hadrons[..., 2]

            # Create masks for non-zero momentum entries
            mask = (px != 0) | (py != 0) | (pz != 0)

            # Calculate transverse momentum (pT)
            pt = np.sqrt(px**2 + py**2)

            # Initialize arrays for the results with zeros
            phi = np.zeros_like(pt)
            eta = np.zeros_like(pt)

            # Apply the calculations only where the mask is True (non-zero momentum)
            phi[mask] = np.arctan2(py[mask], px[mask])
            theta = np.arctan2(pt[mask], pz[mask])
            eta[mask] = -np.log(np.tan(theta / 2))

            # Define the new angular observable array (pT, phi, eta)
            obs = np.stack([pt, phi, eta], axis=-1)

            # NOTE: Could potentially include prescaling using combined mean and standard deviation

            filenames = np.full(len(exp_obs), filename)
            labels.extend(filenames)

            exp_obs = torch.Tensor(exp_obs.copy())

            data.append(exp_obs)

        data = torch.cat(data, dim=0)


        if self.shuffle == True:
            # Generate a random permutation of indices
            indices = np.random.permutation(len(data))

            # Shuffle both data and labels using the same indices
            data = data[indices]  # Works because data is a tensor
            labels = labels[indices]  # Works because labels is a NumPy array
            
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state)
            
            return X_train, X_test, y_train, y_test

        
        
        else:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test

            
        
    def create_sets(self):
        # sets = np.zeros(self.data_paths_len * self.n_datasets, )
        train_data, test_data, train_labels, test_labels = self.prepare_datasets(self.data_paths)


        return {
            "datasets": train_data,
            "labels": np.array(labels),

        }
