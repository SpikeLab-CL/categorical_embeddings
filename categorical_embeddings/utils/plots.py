from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pandas as pd

def plot_embeddings(components=None, dims="2d"):
    """Plots embeddings projections using PCA
        Arguments:
            components: pd.DataFrame() With embeddings components and category in the last column
            dims: <string> Projection, could be "2d" or "3d"
        Returns:
            Plot: Plot with the embeddings projection
    """
    assert (dims=="2d" or dims=="3d"), "dims: must be 2d or 3d"
    assert components.dtypes[-1] == np.dtype(object), "Last column must be the category"
    values = components.iloc[:, 0:len(components.columns)-1]
    categories = components.iloc[:,len(components.columns)-1]
    if dims == "2d":
        pca = PCA(n_components=2)    
        pca_results = pca.fit_transform(values)
        f = pd.DataFrame(pca_results)
        f.columns = ['PCA1','PCA2']
        pca = PCA(n_components=2)    
        pca_results = pca.fit_transform(values)
        f = pd.DataFrame(pca_results)
        f.columns = ['PCA1','PCA2']
        n = list(categories)
        xs = f['PCA1']
        ys = f['PCA2']
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        for index, e in f.iterrows():
            x = e['PCA1']
            y = e['PCA2']
            ax.scatter(x, y, color='b')
            ax.text(x, y, '%s' % (n[index]), size=9, zorder=1, color='k')
        plt.draw()
    else:
        pca = PCA(n_components=3)    
        pca_results = pca.fit_transform(values)
        f = pd.DataFrame(pca_results)
        f.columns = ['PCA1','PCA2','PCA3']
        n = list(categories)
        xs = f['PCA1']
        ys = f['PCA2']
        zs = f['PCA3']
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection='3d')
        for index, e in f.iterrows():
            x = e['PCA1']
            y = e['PCA2']
            z = e['PCA3']
            ax.scatter(x, y, z, color='b')
            ax.text(x, y, z, '%s' % (n[index]), size=9, zorder=1, color='k')
        plt.draw()