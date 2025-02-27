import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

def shift_towards_nearest_other_class(X, y, X_ref, y_ref, **kw):
    pca = PCA(n_components=kw['n_components'], whiten=False)
    X_pca = pca.fit_transform(X)
    X_ref_pca = pca.transform(X_ref)

    centre = np.array([X_ref_pca[y_ref == label].mean(axis=0) for label in sorted(np.unique(y_ref))])

    tree = KDTree(centre)
    distances, indices = tree.query(X_pca, k=2)  # Shape (N, 2)
    nearest_classes = indices[np.arange(len(y)), (indices[:, 0] == y).astype(int)]

    W = pca.components_.T
    W_pinv = np.linalg.pinv(W)
    delta_X = (centre[nearest_classes] - X_pca) @ W_pinv  # Compute shift in original space

    linf_norms = np.max(np.abs(delta_X), axis=1, keepdims=True)  # Compute L∞ norm of each vector
    return delta_X * np.minimum(1, kw['epsilon']/ linf_norms)  # Compute scaling factors




