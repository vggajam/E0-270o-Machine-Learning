import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        # K-means++ : choose one random datapoint as a cluster center and incrementally find other farther points as other cluster centers
        self.cluster_centers = np.array([X[np.random.randint(len(X))],])
        while len(self.cluster_centers) < self.num_clusters:
            distance_from_each_cluster_center = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :], axis=2)
            sum_of_distance_from_each_cluster_center = np.sum(distance_from_each_cluster_center, axis=1)
            max_distant_datapoint_idx = np.argmax(sum_of_distance_from_each_cluster_center)
            self.cluster_centers = np.append(self.cluster_centers, X[max_distant_datapoint_idx].reshape(1,-1), axis=0)

        for _ in range(max_iter):
            # Assign each sample to the closest prototype
            labels = self.predict(X)

            # Update prototypes
            new_centers = []
            for i in range(self.num_clusters):
                dps = X[labels == i]
                if len(dps):
                    new_mean_pt = np.mean(dps , axis=0)
                    new_centers.append(new_mean_pt)
                else:
                    new_centers.append(self.cluster_centers[i])
            new_centers = np.array(new_centers)
            
            if np.linalg.norm(new_centers - self.cluster_centers) < self.epsilon:
                break
            
            self.cluster_centers = new_centers

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        return  np.argmin(np.linalg.norm(X[:,np.newaxis,:]-self.cluster_centers[np.newaxis,:,:], axis=2), axis=1)
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        return self.cluster_centers[self.predict(X)]