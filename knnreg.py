import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MyKNNReg:
    """
    Класс для реализации алгоритма k-ближайших соседей для регрессии.

    Attributes:
        k (int): Количество ближайших соседей.
        metric (str): Метрика для расчета расстояния.
        weight (str): Тип весов для предсказания.
        X_train (pd.DataFrame): Обучающая выборка.
        y_train (pd.Series): Целевая переменная обучающей выборки.
    """

    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        """
        Инициализация класса.

        Args:
            k (int): Количество ближайших соседей.
            metric (str): Метрика для расчета расстояния.
            weight (str): Тип весов для предсказания.
        """
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X_train = None
        self.y_train = None

    def __repr__(self):
        return f'MyKNNReg class: k={self.k}'

    def fit(self, X, y):
        """
        Обучение модели.

        Args:
            X (pd.DataFrame): Обучающая выборка.
            y (pd.Series): Целевая переменная обучающей выборки.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

    def predict(self, X_test):
        """
        Предсказание значений целевой переменной.

        Args:
            X_test (pd.DataFrame): Тестовая выборка.

        Returns:
            np.array: Предсказанные значения целевой переменной.
        """
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
            weight = self._calculate_weight(k_nearest_distances)

            # Нормализуем веса
            weight /= weight.sum()

            # Вычисляем предсказание с учетом весов
            prediction = np.dot(weight, k_nearest_targets)
            predictions.append(prediction)

        return np.array(predictions)

    def _calculate_distance(self, x1, x2):
        """
        Расчет расстояния между двумя точками.

        Args:
            x1 (pd.DataFrame): Первая точка.
            x2 (pd.Series): Вторая точка.

        Returns:
            np.array: Расстояние между двумя точками.
        """
        metric = {
            'euclidean': lambda x1, x2: np.sqrt(((x1 - x2) ** 2).sum(axis=1)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
            'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=1),
            'cosine': lambda x1, x2: 1 - np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2)),
        }
        return metric[self.metric](x1, x2)

    def _calculate_weight(self, distances):
        """
        Расчет весов для предсказания.

        Args:
            distances (np.array): Расстояния до ближайших соседей.

        Returns:
            np.array: Веса для предсказания.
        """
        weight = {
            'distance': 1 / (distances + 1e-12),
            'rank': 1 / (np.arange(1, self.k + 1)),
            'uniform': np.ones(self.k)
        }[self.weight]
        return weight


reg = MyKNNReg(k=1, metric='cosine', weight='rank')
reg.fit(X, y)

y_pred = reg.predict(X)

print(sum(y_pred))
