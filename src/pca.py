from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


class PCAWithEigenDecomposition:

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()

        self.has_run_fit = False
        

    def fit(self, X):
        self.has_run_fit = True
        
        # Getting the dimensions of the data
        n, p = X.shape[0], X.shape[1]
        
        # 1. Mean center the data
        self.mean = X.mean(axis=0)
        X = X - self.mean

        # 2. Compute the Covariance matrix
        cov = np.cov(X, rowvar=False)

        # 3. Eigen value decomposition
        eigvals, eigvectors = np.linalg.eig(cov)

        # Get the top `n_components` eigenvectors based on the eigen values
        sorted_index = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[sorted_index]
        self.eigvectors = eigvectors[:, sorted_index]
        self.eigvectors = self.eigvectors[:, :self.n_components]

        # The variance explained by the individual PCs
        self.explained_variance_ratio = self.eigvals / np.sum(self.eigvals)
        self.explained_variance_ratio = self.explained_variance_ratio[:self.n_components]

        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(1, self.n_components + 1), self.cum_explained_variance, '-o')
        plt.xticks(np.arange(1, self.n_components + 1))
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()
        
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        

    def transform(self, X):
        if not self.has_run_fit:
            raise Exception("Have to call the .fit() method first before calling the .transform() method.")

        # Center the data
        centered_X = X - self.mean
        
        # Project data onto the new feature space
        return np.dot(centered_X, self.eigvectors)


class RandomPCA:

    def __init__(self, n_components: int, top_k: int):
        self.n_components = n_components
        self.top_k = top_k
        self.scaler = StandardScaler()

        self.has_run_fit = False
        

    def fit(self, X):
        self.has_run_fit = True
        
        # Getting the dimensions of the data
        n, p = X.shape[0], X.shape[1]
        
        # 1. Mean center the data
        self.mean = X.mean(axis=0)
        X = X - self.mean

        # 2. Compute the Covariance matrix
        cov = np.cov(X, rowvar=False)

        # 3. Eigen value decomposition
        eigvals, eigvectors = np.linalg.eig(cov)

        # Sorting the eigenvalues and eigenvectors based on the eigen values
        sorted_index = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[sorted_index]
        self.eigvectors = eigvectors[:, sorted_index]

        eigval_sum = np.sum(self.eigvals)
        
        # Getting Random `n_components` eigenvalues and eigenvectors from the first `top_k` eigenvectors
        rand_index = np.random.choice(np.arange(self.top_k), size=self.n_components, replace=False)
        print(f"Rand Index = {rand_index}")
        
        self.eigvals = eigvals[rand_index]
        self.eigvectors = self.eigvectors[:, rand_index]
        
        sorted_index = np.argsort(self.eigvals)[::-1]
        self.eigvals = self.eigvals[sorted_index]
        self.eigvectors = self.eigvectors[:, sorted_index]

        # The variance explained by the individual PCs
        self.explained_variance_ratio = self.eigvals / eigval_sum
        # self.explained_variance_ratio = self.explained_variance_ratio[:self.n_components]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(1, self.n_components + 1), self.cum_explained_variance, '-o')
        plt.xticks(np.arange(1, self.n_components + 1))
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()
        
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        

    def transform(self, X):
        if not self.has_run_fit:
            raise Exception("Have to call the .fit() method first before calling the .transform() method.")

        # Center the data
        centered_X = X - self.mean
        
        # Project data onto the new feature space
        return np.dot(centered_X, self.eigvectors)


class CustomKernelPCA:

    def __init__(self, n_components: int, gamma: Optional[float] = None):
        self.n_components = n_components
        # self.gamma = gamma


    def rbf_kernel(self, X: np.array) -> np.array:
        gamma = 1 / (2 * np.var(X))
        
        similarity = squareform(pdist(X, "sqeuclidean"))
        K = np.exp(-gamma * similarity)

        return K
        

    def fit_transform(self, X):
        # X = X - X.mean(axis=0)
        
        K = self.rbf_kernel(X)

        N = K.shape[0]
        U = np.eye(N) / N

        K = K - np.dot(U, K) - np.dot(K, U) + np.dot(np.dot(U, K), U)

        eigenvals, eigenvecs = np.linalg.eigh(K)
        
        sorted_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sorted_idx]
        eigenvecs = eigenvecs[:, sorted_idx]

        self.explained_variance_ = eigenvals / np.sum(eigenvals)
        
        self.eigenvals = eigenvals[:self.n_components]
        self.eigenvecs = eigenvecs[:, :self.n_components]

        X_proj = np.dot(K, self.eigenvecs)
        
        return X_proj