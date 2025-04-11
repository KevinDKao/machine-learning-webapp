import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def create_mesh_grid(X, margin=0.5, step=0.01):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))
    return xx, yy

def generate_sample_data(n_samples=100, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

class LogisticRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, C=1.0):
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(C=C, random_state=42)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_decision_boundary(self, X):
        xx, yy = create_mesh_grid(X)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        return xx, yy, Z
