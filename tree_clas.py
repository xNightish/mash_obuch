import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MyTreeClf:
    
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        
    def __repr__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'
    
    def entropy(self, y):
        """Calculate entropy of a label array."""
        value_counts = y.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts + 1e-12))  # Добавляем малое значение для избежания логарифма от 0

    def get_best_split(self, X, y):
        best_ig = -np.inf
        best_col_name = None
        best_split_value = None

        for col_name in X.columns:
            unique_values = X[col_name].unique()
            unique_values.sort()  # Сортируем уникальные значения для корректного формирования сплитов

            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i + 1]) / 2  # Находим среднее значение между двумя уникальными значениями

                # Создаем подвыборки
                left_mask = X[col_name] <= split_value
                right_mask = X[col_name] > split_value

                left_y = y[left_mask]
                right_y = y[right_mask]

                # Проверяем, достаточно ли примеров в подвыборках
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # Вычисляем прирост информации
                parent_entropy = self.entropy(y)
                left_entropy = self.entropy(left_y)
                right_entropy = self.entropy(right_y)

                # Прирост информации
                ig = parent_entropy - (len(left_y) / len(y) * left_entropy + len(right_y) / len(y) * right_entropy)

                # Проверяем, является ли текущий сплит лучшим
                if ig > best_ig:
                    best_ig = ig
                    best_col_name = col_name
                    best_split_value = split_value

        return best_col_name, best_split_value, best_ig



# Создаем и тестируем классификатор
clf = MyTreeClf()
col_name, split_value, ig = clf.get_best_split(X, y)
print(f'Best split: Column="{col_name}", Value={split_value}, Information Gain={ig}')
