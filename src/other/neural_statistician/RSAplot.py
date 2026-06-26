import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def scatter_contexts(contexts, labels, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    contexts = np.array(contexts)
    if np.shape(contexts)[1] != 3:
        pca = PCA(n_components=3)
        shape = np.shape(contexts)
        contexts = contexts.reshape(shape[0] * shape[1], shape[2])
        # print(np.shape(contexts))
        # contexts = pca.fit_transform(contexts)
    
        # Optional: Scale the data to have zero mean and unit variance
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(contexts)

        # Apply t-SNE for 2D or 3D visualization
        tsne = TSNE(n_components=3)  # Use 3 for 3D visualization
        contexts = tsne.fit_transform(data_scaled)

    n = len(contexts)
    labels = labels[:n]
    unique_labels = np.unique(labels)
    ix = [np.where(labels == label)
          for i, label in enumerate(unique_labels)]
    colors = [
        'indianred',
        'forestgreen',
        'gold',
        'cornflowerblue',
        'darkviolet'
    ]

    for label, i in enumerate(ix):
        ax.scatter(contexts[i][:, 0], contexts[i][:, 1], contexts[i][:, 2],
                   label=unique_labels[label].title(),
                   color=colors[label])
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    plt.legend(loc='upper left')
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
    plt.close()


def contexts_by_moment(contexts, moments, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    contexts = np.array(contexts).reshape(-1, 3)
    cax = ax.scatter(contexts[:, 0], contexts[:, 1], contexts[:, 2],
                     c=moments[:len(contexts)])
    fig.colorbar(cax)

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
    plt.close()
