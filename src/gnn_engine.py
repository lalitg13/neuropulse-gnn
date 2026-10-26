import numpy as np

class SpatioTemporalGCN:
    """
    Spatio-Temporal Graph Convolutional Network (GCN)
    Implements spectral graph convolution with Kipf-Welling normalization:
    A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
    """
    def __init__(self, adj_matrix, in_features=4, hidden_features=16, out_features=1, lr=0.01):
        np.random.seed(42)
        N = adj_matrix.shape[0]
        self.N = N
        
        # Self-loops & Kipf-Welling Renormalization trick
        A_tilde = adj_matrix + np.eye(N)
        degrees = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_tilde_inv_sqrt = np.diag(d_inv_sqrt)
        self.A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        
        # Two-layer GCN weights
        self.W0 = np.random.randn(in_features, hidden_features) * np.sqrt(2.0 / in_features)
        self.b0 = np.zeros((1, hidden_features))
        self.W1 = np.random.randn(hidden_features, out_features) * np.sqrt(2.0 / hidden_features)
        self.b1 = np.zeros((1, out_features))
        
        self.lr = lr
        self.loss_history = []
        
    def _relu(self, x):
        return np.maximum(0, x)
        
    def _drelu(self, x):
        return (x > 0).astype(float)
        
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
        
    def forward(self, X):
        """
        Forward pass: H1 = ReLU(A_norm * X * W0), Y_hat = Sigmoid(A_norm * H1 * W1)
        """
        # Layer 1
        self.AX = self.A_norm @ X
        self.z0 = self.AX @ self.W0 + self.b0
        self.h1 = self._relu(self.z0)
        
        # Layer 2
        self.Ah1 = self.A_norm @ self.h1
        self.z1 = self.Ah1 @ self.W1 + self.b1
        self.y_hat = self._sigmoid(self.z1)
        return self.y_hat
        
    def train_step(self, X, y_target):
        y_pred = self.forward(X)
        error = y_pred - y_target
        loss = np.mean(error ** 2)
        
        # Backprop through Layer 2
        dy = (2.0 / len(y_target)) * error * (y_pred * (1.0 - y_pred))
        dW1 = self.Ah1.T @ dy
        db1 = np.sum(dy, axis=0, keepdims=True)
        
        # Backprop through Graph Convolution & Layer 1
        dAh1 = dy @ self.W1.T
        dh1 = self.A_norm.T @ dAh1
        dz0 = dh1 * self._drelu(self.z0)
        dW0 = self.AX.T @ dz0
        db0 = np.sum(dz0, axis=0, keepdims=True)
        
        # Gradient updates
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W0 -= self.lr * dW0
        self.b0 -= self.lr * db0
        
        self.loss_history.append(float(loss))
        return loss, y_pred
