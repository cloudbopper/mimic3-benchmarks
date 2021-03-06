"""Model wrapper to analyze MIMIC-3 model using anamod"""

from mimic3ext import ext_utils
import numpy as np

class ModelWrapper():
    """Model wrapper around MIMIC-3 TensorFlow model"""
    def __init__(self, model):
        self.model = model  # Tensorflow Model with loaded weights

    def predict(self, X):
        """Predict output probabilities on instances in X"""
        X = np.transpose(X, (0, 2, 1))  # TF model expects instances X timestamps X features
        return self.model.predict(X)[:, 0]
