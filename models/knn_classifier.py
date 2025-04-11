import numpy as np

def create_mesh_grid(X, margin=0.5, step=0.01):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))
    return xx, yy

def generate_sample_data(n_samples=120, random_state=42):
    """Generate an interesting pattern of points that shows k-sensitivity"""
    np.random.seed(random_state)
    
    # Create class 0 points (three clusters)
    n_per_cluster = n_samples // 6
    
    # Cluster 1 (bottom left)
    X0_1 = np.random.randn(n_per_cluster, 2) * 0.3 + [-2, -1]
    
    # Cluster 2 (middle)
    X0_2 = np.random.randn(n_per_cluster, 2) * 0.3 + [0, -0.5]
    
    # Cluster 3 (top right)
    X0_3 = np.random.randn(n_per_cluster, 2) * 0.3 + [2, 1]
    
    # Create class 1 points (three clusters)
    # Cluster 1 (top left)
    X1_1 = np.random.randn(n_per_cluster, 2) * 0.3 + [-2, 1]
    
    # Cluster 2 (middle)
    X1_2 = np.random.randn(n_per_cluster, 2) * 0.3 + [0, 0.5]
    
    # Cluster 3 (bottom right)
    X1_3 = np.random.randn(n_per_cluster, 2) * 0.3 + [2, -1]
    
    # Combine all points
    X = np.vstack([X0_1, X0_2, X0_3, X1_1, X1_2, X1_3])
    y = np.hstack([np.zeros(n_per_cluster * 3), np.ones(n_per_cluster * 3)])
    
    # Add some noise points to make it more interesting
    noise_points = n_samples // 6
    X_noise = np.random.uniform(-3, 3, (noise_points, 2))
    y_noise = (np.random.rand(noise_points) > 0.5).astype(float)
    
    X = np.vstack([X, X_noise])
    y = np.hstack([y, y_noise])
    
    # Shuffle the data
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

class KNNClassifierModel:
    def __init__(self):
        self.X = None
        self.y = None
        self.n_neighbors = 5
        
    def fit(self, X, y, n_neighbors=5):
        """Store the data and n_neighbors"""
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
    
    def get_decision_boundary(self, X):
        """Create a smooth simulated decision boundary"""
        # Create mesh grid
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Create a smooth decision boundary that looks like KNN
        k = self.n_neighbors
        
        # Create base pattern
        Z = np.zeros_like(xx)
        
        # Add circular influence around each point
        for x, y, label in zip(self.X[:, 0], self.X[:, 1], self.y):
            # Calculate distance influence (gaussian-like)
            dist = ((xx - x)**2 + (yy - y)**2)
            # Make the influence more local for small k and more spread out for large k
            spread = 2.0 / (k + 1)  # Adjust spread based on k
            influence = np.exp(-dist / spread)
            if label == 1:
                Z += influence
            else:
                Z -= influence
        
        # Normalize and threshold
        Z = (Z > 0).astype(float)
        
        # Add some noise to make it look more realistic
        noise = np.random.randn(*Z.shape) * 0.05  # Reduced noise
        Z = ((Z + noise) > 0.5).astype(float)
        
        return xx, yy, Z
