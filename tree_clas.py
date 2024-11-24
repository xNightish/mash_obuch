import numpy as np
import pandas as pd
from graphviz import Digraph

# Загружаем данные
df = pd.read_csv('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:, :4], df['target']

class MyTreeClf:
    
    def __init__(self, max_depth=15, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.leafs_cnt = 0 
        self.tree = None
        self.histograms = {}
        self.criterion = criterion
        self.reserved_leaves = 0

    def entropy(self, y):
        value_counts = y.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))
    
    def gini(self, y):
        value_counts = y.value_counts(normalize=True)
        return 1 - np.sum(value_counts**2)
    
    
    def impurity(self, y):
        criterion = {'entropy': self.entropy, 'gini': self.gini}
        return criterion[self.criterion](y)

    def get_best_split(self, X, y):
        best_ig = -np.inf
        best_col_name = None
        best_split_value = None
        parent_impurity = self.impurity(y)

        for col_name in X.columns:
            unique_values = X[col_name].unique()
            unique_values.sort()

            if self.bins is not None:
                if col_name not in self.histograms:
                    hist, bin_edges = np.histogram(X[col_name], bins=self.bins)
                    self.histograms[col_name] = bin_edges[1:-1]  # Убираем крайние значения
                split_values = self.histograms[col_name]
            else:
                split_values = unique_values[:-1] + np.diff(unique_values) / 2  # Обычные разделители

            # Проверяем, есть ли разделители
            if len(split_values) == 0:
                continue

            for split_value in split_values:
                left_mask = X[col_name] <= split_value
                right_mask = X[col_name] > split_value

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                left_impurity = self.impurity(left_y)
                right_impurity = self.impurity(right_y)

                ig = parent_impurity - (len(left_y) / len(y) * left_impurity + len(right_y) / len(y) * right_impurity)

                if ig > best_ig:
                    best_ig = ig
                    best_col_name = col_name
                    best_split_value = split_value

        return best_col_name, best_split_value, best_ig
    
    def fit(self, X, y, depth=0, node=None):
        if node is None:
            self.tree = {}
            node = self.tree
            is_root = True
        else:
            is_root = False

        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1 or 
            self.max_leafs + 1 - self.leafs_cnt - self.reserved_leaves <= 0) and not is_root:
            node['leaf'] = y.value_counts(normalize=True).to_dict()
            self.leafs_cnt += 1
            return

        col_name, split_value, ig = self.get_best_split(X, y)

        if col_name is None:
            node['leaf'] = y.value_counts(normalize=True).to_dict()
            self.leafs_cnt += 1
            return

        node[col_name] = {}
        node[col_name][f'<= {split_value}'] = {}
        node[col_name][f'> {split_value}'] = {}

        self.reserved_leaves += 1

        left_mask = X[col_name] <= split_value
        right_mask = X[col_name] > split_value

        self.fit(X[left_mask], y[left_mask], depth + 1, node[col_name][f'<= {split_value}'])
        self.fit(X[right_mask], y[right_mask], depth + 1, node[col_name][f'> {split_value}'])
        
        self.reserved_leaves -= 1


    def visualize_tree(self, node=None, graph=None, parent=None, edge_label=''):
        if graph is None:
            graph = Digraph()

        if node is None:
            node = self.tree

        for key, value in node.items():
            if key == 'leaf':
                leaf_str = ', '.join([f'{k}: {v:.2f}' for k, v in value.items()])
                graph.node(str(id(node)), f'Leaf: {leaf_str}', shape='box', style='filled', fillcolor='lightgreen')  # Лист зеленый
                if parent is not None:
                    # Цвет стрелки определяется по edge_label
                    edge_color = 'blue' if '<=' in edge_label else 'red'
                    graph.edge(parent, str(id(node)), label=edge_label, color=edge_color)  # Стрелка с цветом
            else:
                split_value = list(value.keys())[0].split()[1]
                left_node = value[f'<= {split_value}']
                right_node = value[f'> {split_value}']

                # Устанавливаем цвет узла в зависимости от вероятностей
                left_leaf_probability = left_node.get('leaf', {}).get(1, 0)
                right_leaf_probability = right_node.get('leaf', {}).get(1, 0)

                # Устанавливаем цвет узла (ветви) и стрелки
                if parent is None:
                    fill_color = '#FFA500'  # Оранжевый для корня
                else:
                    fill_color = '#FFCCCB' if left_leaf_probability > right_leaf_probability else '#ADD8E6'  # Красный или голубой для ветвей

                graph.node(str(id(value)), f'{key} <= {split_value}', style='filled', fillcolor=fill_color)

                if parent is not None:
                    # Цвет стрелки определяется на основе edge_label
                    edge_color = 'blue' if '<=' in edge_label else 'red'
                    graph.edge(parent, str(id(value)), label=edge_label, color=edge_color)

                self.visualize_tree(left_node, graph, str(id(value)), '<=' + split_value)
                self.visualize_tree(right_node, graph, str(id(value)), '>' + split_value)

        return graph

    def _class(self, row):
        node = self.tree
        while True:
            if 'leaf' in node:
                return node['leaf'].get(1, 0) 
            if not node or not node.keys():
                return 0 
            col_name = list(node.keys())[0]
            if col_name not in node:
                return 0 
            split_value = list(node[col_name].keys())[0].split()[1]
            if row[col_name] <= float(split_value):
                node = node[col_name][f'<= {split_value}']
            else:
                node = node[col_name][f'> {split_value}']

    def predict_proba(self, X):
        return X.apply(self._class, axis=1)

    def predict(self, X):
        return self.predict_proba(X).apply(lambda x: 1 if x > 0.5 else 0)
    
    def sum_leaf_probabilities(self):
        return self._sum_leaf_probabilities(self.tree)

    def _sum_leaf_probabilities(self, node):
        if 'leaf' in node:
            return node['leaf'].get(1, 0)  # Вернуть вероятность класса 1 в листе
        else:
            total_probability = 0
            for key, value in node.items():
                total_probability += self._sum_leaf_probabilities(value)
            return total_probability
    
    
# Создаем и тестируем классификатор
clf = MyTreeClf(max_depth=10, min_samples_split=40, max_leafs=21, bins=10, criterion='gini')
clf.fit(X, y)

# Визуализируем дерево
tree_graph = clf.visualize_tree()
tree_graph.render('decision_tree', format='png', cleanup=True)  # Сохраняем в формате PNG

print(clf.leafs_cnt)

# Предсказываем вероятности классов
y_proba = clf.predict(X)

# Проверяем точность классификатора
accuracy = (y_proba == y).mean()
print(f'Accuracy: {accuracy}')

# Получаем сумму вероятностей класса 1 на всех листьях
total_probability = clf.sum_leaf_probabilities()
print(f'Total probability of class 1 on leaves: {total_probability}')
