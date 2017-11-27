import numpy as np


class KMeans:
    def __init__(self):
        self._centroids = None
        self.mean = None
        self.std = None

    @property
    def centroids(self):
        return self.denormalize(self._centroids)

    def fit(self, X, K, epochs):
        cid = None
        X, self.mean, self.std = KMeans._normalize_and_get_params(X)
        self._centroids = KMeans._select_random_instances(X, K)
        for e in np.arange(epochs):
            last_centroids = self._centroids.copy()
            cid = KMeans._assign_cluster(X, self._centroids)
            self._centroids = KMeans._update_centroids(X, cid, K)
            if np.array_equal(last_centroids, self._centroids):
                break

    @staticmethod
    def _normalize_and_get_params(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std, mean, std

    def _normalize(self, X):
        return (X - self.mean) / self.std

    def denormalize(self, X):
        return (X * self.std) + self.mean

    @staticmethod
    def _select_random_instances(X, K):
        T = X.copy()
        np.random.permutation(T)
        return T[:K]

    @staticmethod
    def _assign_cluster(X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    @staticmethod
    def _update_centroids(X, cid, K):
        return np.array([X[cid == k].mean(axis=0) for k in range(K)])

    def predict(self, X):
        X = self._normalize(X)
        return self._assign_cluster(X, self._centroids)