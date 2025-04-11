import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def generate_sample_data(n_samples=40, random_state=42):
    """Generate a more dramatic curve with sparser points"""
    np.random.seed(random_state)
    # Create non-uniform X points to show interesting regions
    X = np.concatenate([
        np.random.uniform(-3, -1, n_samples//4),  # Cluster in negative region
        np.random.uniform(-0.5, 0.5, n_samples//4),  # Cluster near zero
        np.random.uniform(1, 3, n_samples//2),  # Cluster in positive region
    ]).reshape(-1, 1)
    X = np.sort(X, axis=0)
    
    # Create a more dramatic curve with multiple features
    y = 0.5 * np.sin(2 * X.ravel()) + 0.3 * np.cos(5 * X.ravel()) + \
        np.random.normal(0, 0.2, n_samples)  # Add some noise
    
    return X, y

class KNNRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, n_neighbors=5):
        X_scaled = self.scaler.fit_transform(X)
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_prediction_line(self, X):
        X_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 200).reshape(-1, 1)
        y_pred = self.predict(X_line)
        return X_line, y_pred
