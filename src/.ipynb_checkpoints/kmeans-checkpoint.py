import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances


class CustomKMeans:
      
    def __init__(self, n_clusters: int, max_iters: int = 500, tolerance: float = 1e-2):
        self.n_clusters = n_clusters

        self.max_iters = max_iters
        self.tolerance = tolerance
        
        
    @staticmethod
    def euclidean_distance(points, centroids) -> np.array:
        distances = np.empty(shape=(centroids.shape[0], points.shape[0]))

        chunk_size = 2000
        idx = 0

        for idx in range(0, points.shape[0], chunk_size):
            points_subset = points[idx : idx + chunk_size, :]

            distances[:, idx : idx + chunk_size] = euclidean_distances(centroids, points_subset)

            idx += chunk_size

        return distances
        

    def e_step(self, X_train: np.array, mu_k: np.array) -> np.array:
        distances = self.euclidean_distance(X_train, mu_k)

        clusters = np.argmin(distances, axis=0)

        pi_ik = np.zeros(shape=(X_train.shape[0], self.n_clusters))
        pi_ik[np.arange(X_train.shape[0]), clusters] = 1

        return pi_ik


    def m_step(self, X_train: np.array, pi_ik: np.array) -> np.array:
        mu_k = np.empty(shape=(self.n_clusters, X_train.shape[1]))

        for cluster_idx in range(self.n_clusters):
            current_cluster_points = X_train[np.where(pi_ik[:, cluster_idx] == 1)]
            mu_k[cluster_idx, :] = np.mean(current_cluster_points, axis=0)

        return mu_k


    def fit(self, X_train: np.array) -> None:
        random_centroids_idx = np.random.choice(
            np.arange(X_train.shape[0]), size=self.n_clusters, replace=False
        )

        mu_k = X_train[random_centroids_idx]

        for _ in tqdm(range(self.max_iters), desc="Iterations"):
            pi_ik = self.e_step(X_train, mu_k)

            prev_mu_k = mu_k

            mu_k = self.m_step(X_train, pi_ik)

            if np.linalg.norm(prev_mu_k - mu_k) < self.tolerance:
                break
        
        self.mu_k = mu_k


    def predict(self, X: np.array):
        distances = self.euclidean_distance(X, self.mu_k)
        clusters = np.argmin(distances, axis=0)

        return clusters
    

    @staticmethod
    def get_confusion_matrix(n_true_clusters, n_algo_clusters, y_true, y_pred):
        cm = np.zeros(shape=(n_true_clusters, n_algo_clusters), dtype=int)

        ab = np.squeeze(np.rec.fromarrays([y_true, y_pred]))
        unique, counts = np.unique(ab, return_counts=True, axis=0)

        for idx in range(unique.shape[0]):
            cm[unique[idx][0], unique[idx][1]] = counts[idx]

        return cm
    

    @staticmethod
    def purity_metric(cm: np.array) -> float:
        return np.round(np.sum(np.amax(cm, axis=0)) / np.sum(cm), 3)
    

    @staticmethod
    def avg_gini_coefficient(cm: np.array) -> float:
        M_j = np.sum(cm, axis=0)
        G_j = np.empty_like(M_j, dtype=float)

        for idx in range(M_j.shape[0]):
            G_j[idx] = 1 - np.sum(np.power(cm[:, idx] / M_j[idx], 2))

        return np.round(np.dot(G_j, M_j.T) / np.sum(M_j), 3)