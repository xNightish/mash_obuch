import pandas as pd
import numpy as np

class MyKNNClf:
    def __init__(self, k: int = 1, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X_train = None
        self.y_train = None 
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self 
    
    
    def get_dist(self, row):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train.values - row.values) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train.values - row.values), axis=1)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train.values - row.values), axis=1)
        elif self.metric == 'cosine':
            norm_X = np.linalg.norm(self.X_train.values, axis=1)
            norm_row = np.linalg.norm(row.values)
            return 1 - np.dot(self.X_train.values, row.values) / (norm_X * norm_row)
        else:
            return None

    def _class(self, row):
        distance = self.get_dist(row)
        sorted_indices = np.argsort(distance)[:self.k]
        neighbors_classes = self.y_train.iloc[sorted_indices]

        if self.weight == 'uniform':
            weights = np.ones(self.k)
        elif self.weight == 'rank':
            weights = 1 / (np.arange(1, self.k + 1))
        elif self.weight == 'distance':
            weights = 1 / (distance[sorted_indices])  # Избегаем деления на ноль
        else:
            return None

        # Суммируем веса классов 
        class_weights = {}
        for cls in np.unique(neighbors_classes):
            class_weights[cls] = np.sum(weights[neighbors_classes == cls])

        # Возвращаем класс с наибольшим весом
        return max(class_weights, key=class_weights.get)

    def _proba(self, row):
        distance = self.get_dist(row)
        sorted_indices = np.argsort(distance)[:self.k]
        neighbors_classes = self.y_train.iloc[sorted_indices]

        if self.weight == 'uniform':
            weights = np.ones(self.k)
        elif self.weight == 'rank':
            weights = 1 / (np.arange(1, self.k + 1))
        elif self.weight == 'distance':
            weights = 1 / (distance[sorted_indices])  # Избегаем деления на ноль 
        else:
            return None

        # Суммируем веса классов
        class_weights = {}
        for cls in np.unique(neighbors_classes):
            class_weights[cls] = np.sum(weights[neighbors_classes == cls])

        # Возвращаем нормализованный вес класса 1
        total_weight = np.sum(weights)
        return class_weights.get(1, 0) / total_weight if total_weight > 0 else 0

    def predict(self, X: pd.DataFrame):
        return X.apply(self._class, axis=1)

    def predict_proba(self, X: pd.DataFrame):
        return X.apply(self._proba, axis=1)

    def __repr__(self):
        atts = ', '.join([f'{k}={v}' for k, v in vars(self).items()])
        return f'MyKNNClf class: {atts}'
    
    
        
df = pd.read_csv('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']


knn = MyKNNClf(k=5, weight='rank')
knn.fit(X, y)
        
predictions = knn.predict(X)
probabilities = knn.predict_proba(X)
prob_sum = round(probabilities.sum(), 10)
print(f"Предсказания: {predictions}\n")

