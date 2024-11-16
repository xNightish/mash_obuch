import numpy as np
import pandas as pd


class MyKNNReg:
    def __init__(self, k = 3):
        self.k = k
        self.train_size = 0
        
    def __repr__(self):
        return f'MyKNNReg class: k={self.k}'
    
    
    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.train_size = X.shape
        return self.train_size