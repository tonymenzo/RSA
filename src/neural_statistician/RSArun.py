import argparse
import os
import time

from RSAdata import RSA_Dataset, prepare_datasets
from RSAmodel import Statistician
from RSAplot import scatter_contexts, contexts_by_moment
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

import torch
import json

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Using device: ', device)

# command line args
# parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')

# required
# parser.add_argument('--output-dir', required=True, type=str, default=None,
#                     help='output directory for checkpoints and figures')

# # optional
# parser.add_argument('--n-datasets', type=int, default=10000, metavar='N',
#                     help='number of synthetic datasets in collection (default: 10000)')
# parser.add_argument('--batch-size', type=int, default=16,
#                     help='batch size (of datasets) for training (default: 16)')
# parser.add_argument('--sample-size', type=int, default=200,
#                     help='number of samples per dataset (default: 200)')
# parser.add_argument('--n-features', type=int, default=1,
#                     help='number of features per sample (default: 1)')
# parser.add_argument('--distributions', type=str, default='easy',
#                     help='which distributions to use for synthetic data '
#                          '(easy: (Gaussian, Uniform, Laplacian, Exponential), '
#                          'hard: (Bimodal mixture of Gaussians, Laplacian, '
#                          'Exponential, Reverse Exponential) '
#                          '(default: easy)')
# parser.add_argument('--c-dim', type=int, default=3,
#                     help='dimension of c variables (default: 3)')
# parser.add_argument('--n-hidden-statistic', type=int, default=3,
#                     help='number of hidden layers in statistic network modules '
#                          '(default: 3)')
# parser.add_argument('--hidden-dim-statistic', type=int, default=128,
#                     help='dimension of hidden layers in statistic network (default: 128)')
# parser.add_argument('--n-stochastic', type=int, default=1,
#                     help='number of z variables in hierarchy (default: 1)')
# parser.add_argument('--z-dim', type=int, default=32,
#                     help='dimension of z variables (default: 32)')
# parser.add_argument('--n-hidden', type=int, default=3,
#                     help='number of hidden layers in modules outside statistic network '
#                          '(default: 3)')
# parser.add_argument('--hidden-dim', type=int, default=128,
#                     help='dimension of hidden layers in modules outside statistic network '
#                          '(default: 128)')
# parser.add_argument('--print-vars', type=bool, default=False,
#                     help='whether to print all learnable parameters for sanity check '
#                          '(default: False)')
# parser.add_argument('--learning-rate', type=float, default=1e-3,
#                     help='learning rate for Adam optimizer (default: 1e-3).')
# parser.add_argument('--epochs', type=int, default=50,
#                     help='number of epochs for training (default: 50)')
# parser.add_argument('--viz-interval', type=int, default=-1,
#                     help='number of epochs between visualizing context space '
#                          '(default: -1 (only visualize last epoch))')
# parser.add_argument('--save_interval', type=int, default=-1,
#                     help='number of epochs between saving model '
#                          '(default: -1 (save on last epoch))')
# parser.add_argument('--clip-gradients', type=bool, default=True,
#                     help='whether to clip gradients to range [-0.5, 0.5] '
#                          '(default: True)')
# args = parser.parse_args()

output_dir = 'output'
epochs = 200
viz_interval = 5
save_interval = -1
clip_gradients = True
batch_size = 32
learning_rate = 1e-3

# sample_size = 200
n_features = 3
c_dim = 128
n_hidden_statistic = 3
hidden_dim_statistic = 256
n_stochastic = 3
z_dim = 128
n_hidden = 3
hidden_dim = 256
print_vars = False

assert output_dir is not None
os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")

viz_interval = epochs if viz_interval == -1 else viz_interval
save_interval = epochs if save_interval == -1 else save_interval

# New figure directory path
directory_path = os.path.join(output_dir, 'figures', time_stamp)

# Ensure the directory exists
os.makedirs(directory_path, exist_ok=True)

def run(model, optimizer, loaders, datasets, model_kwargs_str):

    train_loader, test_loader = loaders
    train_dataset, test_dataset = datasets


    alpha = 1
    tbar = tqdm(range(epochs))
    # main training loop
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        for batch in train_loader:
            vlb = model.step(batch, alpha, optimizer, clip_gradients=clip_gradients)
            running_vlb += vlb

        running_vlb /= (len(train_dataset) // batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # show test set in context space at intervals
        if (epoch + 1) % viz_interval == 0:
            model.eval()
            contexts = []
            for batch in test_loader:
                with torch.no_grad():
                    inputs = batch.to(device)
                context_means, _ = model.statistic_network(inputs)
                contexts.append(context_means.data.cpu().numpy())

            # show coloured by distribution
            path = directory_path + '/{}.png'.format(epoch + 1)
            scatter_contexts(contexts, test_dataset.data['labels'], savepath=path)

            path_kwargs = directory_path + '/model_kwargs.json'
            with open(path_kwargs, 'w') as f:
                f.write(model_kwargs_str)

            # # show coloured by mean
            # path = output_dir + '/figures/' + time_stamp \
            #        + '-{}-mean.pdf'.format(epoch + 1)
            # contexts_by_moment(contexts, moments=test_dataset.data['means'],
            #                    savepath=path)

            # # show coloured by variance
            # path = output_dir + '/figures/' + time_stamp \
            #        + '-{}-variance.pdf'.format(epoch + 1)
            # contexts_by_moment(contexts, moments=test_dataset.data['variances'],
            #                    savepath=path)

        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            save_path = output_dir + '/checkpoints/' + time_stamp \
                        + '-{}.m'.format(epoch + 1)
            model.save(optimizer, save_path)


def main():

    data_paths = ['../../data/structured_data/pgun_qqbar_hadrons_a_0.68_b_0.98_sigma_0.335_N_1e4.npy', 
    '../../data/structured_data/pgun_qqbar_hadrons_a_0.72_b_0.88_sigma_0.335_N_1e4.npy']
    # data_paths = ['/Users/lukapuslar/Desktop/other/ASEF/Project/Code/RSA/RSA/data/structured_data/pgun_qqbar_hadrons_a_0.68_b_0.98_sigma_0.335_N_1e4.npy', 
    # '/Users/lukapuslar/Desktop/other/ASEF/Project/Code/RSA/RSA/data/structured_data/pgun_qqbar_hadrons_a_0.72_b_0.88_sigma_0.335_N_1e4.npy']

    X_train, X_test, y_train, y_test, shape = prepare_datasets(data_paths, test_size= 0.2)

    data_size = shape[0]
    sample_size = shape[1]
    n_features = shape[2]

    train_dataset = RSA_Dataset(X_train, y_train)
    test_dataset = RSA_Dataset(X_test, y_test)


    datasets = (train_dataset, test_dataset)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=0, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0, drop_last=True)
    
    loaders = (train_loader, test_loader)

    model_kwargs = {
        'batch_size': batch_size,
        'sample_size': sample_size,
        'n_features': n_features,
        'c_dim': c_dim,
        'n_hidden_statistic': n_hidden_statistic,
        'hidden_dim_statistic': hidden_dim_statistic,
        'n_stochastic': n_stochastic,
        'z_dim': z_dim,
        'n_hidden': n_hidden,
        'hidden_dim': hidden_dim,
        'nonlinearity': F.relu,
        'print_vars': print_vars,
        'device': device
    }

    model_kwargs_str = model_kwargs.copy()
    model_kwargs_str['nonlinearity'] = 'F.relu'
    model_kwargs_str['device'] = str(device)
    model_kwargs_str = json.dumps(model_kwargs_str, indent=4)


    model = Statistician(**model_kwargs)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run(model, optimizer, loaders, datasets, model_kwargs_str)


if __name__ == '__main__':
    main()
