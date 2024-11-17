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
    
    def predict(self, X_test):
        predictions = []
        
        for index, test_point in X_test.iterrows():
            # Вычисляем расстояние до каждого объекта из обучающей выборки
            distances = np.sqrt(((self.X_train - test_point) ** 2).sum(axis=1))
            
            # Находим индексы k ближайших соседей
            k_indices = distances.nsmallest(self.k).index
            
            # Усредняем значения таргета ближайших k объектов
            k_nearest_targets = self.y_train.iloc[k_indices]
            prediction = k_nearest_targets.mean()
            predictions.append(prediction)
        
        return np.array(predictions)
    