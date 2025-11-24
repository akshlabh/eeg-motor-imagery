# visualize.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_embeddings(embeddings, labels, title="EEG Embeddings (2D PCA)"):
    """
    2D PCA plot of embeddings. labels can be integer class labels.
    """
    embeddings = np.asarray(embeddings)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='rainbow', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()


def visualize_query_result(query_embedding, embeddings, labels, top_indices=None, title="Query vs Stored Embeddings"):
    """
    Visualize query point among stored embeddings. Optionally highlight nearest neighbor indices.
    top_indices: list of indices (ints) to highlight as nearest neighbors
    """
    embeddings = np.asarray(embeddings)
    all_emb = np.vstack([embeddings, query_embedding.reshape(1, -1)])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_emb)

    stored = reduced[:-1]
    query = reduced[-1]

    plt.figure(figsize=(8,6))
    plt.scatter(stored[:,0], stored[:,1], c=labels, cmap='rainbow', alpha=0.6, label='Stored')
    plt.scatter(query[0], query[1], c='black', marker='X', s=120, label='Query')

    if top_indices:
        pts = stored[top_indices]
        plt.scatter(pts[:,0], pts[:,1], c='none', edgecolor='k', s=150, linewidths=1.5, label='Neighbors')

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_3d_tsne(embeddings, labels, title="3D t-SNE Embeddings"):
    """
    3D t-SNE visualization.
    """
    embeddings = np.asarray(embeddings)
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, init='random')
    reduced = tsne.fit_transform(embeddings)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=labels, cmap='rainbow', alpha=0.8)
    fig.colorbar(p, ax=ax)
    ax.set_title(title)
    plt.show()