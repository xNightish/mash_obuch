import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight
        
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
            distances = self._calculate_distance(self.X_train, test_point)
            
            # Находим индексы k ближайших соседей
            k_indices = np.argsort(distances)[:self.k]
            
            # Усредняем значения таргета ближайших k объектов
            k_nearest_targets = self.y_train.iloc[k_indices]
            k_nearest_distances = distances[k_indices]
            
            # Вычисляем веса
            weight = {'distance': 1 / (k_nearest_distances + 1e-12),
                            'rank': 1 / (np.arange(1, self.k + 1)), 
                            'uniform': np.ones(self.k)}[self.weight]
            
            # Нормализуем веса
            weight /= weight.sum()
            
            # Вычисляем предсказание с учетом весов
            prediction = np.dot(weight, k_nearest_targets)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _calculate_distance(self, x1, x2):
        metric = {
            'euclidean': lambda x1, x2: np.sqrt(((x1 - x2) ** 2).sum(axis=1)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
            'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=1),
            'cosine': lambda x1, x2: 1 - np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2)),
        }
        return metric[self.metric](x1, x2)
    
reg = MyKNNReg(k=1, metric='cosine', weight='rank')
reg.fit(X, y)

y_pred = reg.predict(X)

print(sum(y_pred))
