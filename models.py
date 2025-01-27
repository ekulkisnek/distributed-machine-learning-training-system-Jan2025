from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator
import logging

class BaseModel(BaseEstimator):
    def __init__(self):
        self.parameters = {}
        self.logger = logging.getLogger(__name__)
        
    def compute_gradients(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for model parameters"""
        raise NotImplementedError
        
    def update_parameters(self, gradients: Dict[str, np.ndarray], 
                         learning_rate: float = 0.01):
        """Update model parameters using gradients"""
        for param_name, gradient in gradients.items():
            self.parameters[param_name] -= learning_rate * gradient
            
class RidgeRegression(BaseModel):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.parameters['weights'] = None
        self.parameters['bias'] = None
        self.alpha = alpha
        
    def initialize_parameters(self, n_features: int):
        self.parameters['weights'] = np.zeros(n_features)
        self.parameters['bias'] = 0.0
        
    def compute_gradients(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        X, y = data[:, :-1], data[:, -1]
        predictions = np.dot(X, self.parameters['weights']) + self.parameters['bias']
        error = predictions - y
        
        gradients = {
            'weights': (np.dot(X.T, error) / len(X)) + (self.alpha * self.parameters['weights']),
            'bias': np.mean(error)
        }
        return gradients

class LassoRegression(BaseModel):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.parameters['weights'] = None
        self.parameters['bias'] = None
        self.alpha = alpha
        
    def initialize_parameters(self, n_features: int):
        self.parameters['weights'] = np.zeros(n_features)
        self.parameters['bias'] = 0.0
        
    def compute_gradients(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        X, y = data[:, :-1], data[:, -1]
        predictions = np.dot(X, self.parameters['weights']) + self.parameters['bias']
        error = predictions - y
        
        gradients = {
            'weights': (np.dot(X.T, error) / len(X)) + (self.alpha * np.sign(self.parameters['weights'])),
            'bias': np.mean(error)
        }
        return gradients

class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__()
        self.parameters['weights'] = None
        self.parameters['bias'] = None
        
    def initialize_parameters(self, n_features: int):
        """Initialize model parameters"""
        self.parameters['weights'] = np.zeros(n_features)
        self.parameters['bias'] = 0.0
        
    def compute_gradients(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for linear regression"""
        X, y = data[:, :-1], data[:, -1]
        predictions = np.dot(X, self.parameters['weights']) + self.parameters['bias']
        error = predictions - y
        
        gradients = {
            'weights': np.dot(X.T, error) / len(X),
            'bias': np.mean(error)
        }
        return gradients
