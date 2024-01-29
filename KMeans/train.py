from sklearn import datasets
from KMeans import KMeans
import numpy as np

X ,y = datasets.make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=1.0, random_state=42)
print(X.shape)
clusters = len(np.unique(y))
print(clusters)

k = KMeans(k=clusters, max_iter=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()