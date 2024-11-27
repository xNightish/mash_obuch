import numpy as np
import pandas as pd

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        
    def __repr__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'
    
    def get_best_split(self, X, y):
        best_gain = -np.inf
        best_col_name = None
        best_split_value = None
        
        for col_name in X.columns:
            unique_values = np.sort(X[col_name].unique())
            
            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i + 1]) / 2
                gain = self._calculate_mse_gain(X, y, col_name, split_value)
                
                if gain > best_gain:
                    best_gain = gain
                    best_col_name = col_name
                    best_split_value = split_value
        
        return best_col_name, best_split_value, best_gain
    
    def _calculate_mse_gain(self, X, y, col_name, split_value):
        left_mask = X[col_name] <= split_value
        right_mask = X[col_name] > split_value
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf
        
        mse_left = np.mean((y_left - np.mean(y_left)) ** 2)
        mse_right = np.mean((y_right - np.mean(y_right)) ** 2)
        total_mse = (len(y_left) * mse_left + len(y_right) * mse_right) / (len(y_left) + len(y_right))
        
        total_mse_original = np.mean((y - np.mean(y)) ** 2)
        gain = total_mse_original - total_mse
        
        return gain
