import numpy as np
import pandas as pd
from graphviz import Digraph

# Загружаем данные
df = pd.read_csv('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:, :4], df['target']

class MyTreeClf:
    
    def __init__(self, max_depth=15, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0  # Счетчик листьев
        self.tree = None  # Дерево решений

    def entropy(self, y):
        value_counts = y.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts + 1e-12))

    def get_best_split(self, X, y):
        best_ig = -np.inf
        best_col_name = None
        best_split_value = None
        parent_entropy = self.entropy(y)

        for col_name in X.columns:
            unique_values = X[col_name].unique()
            unique_values.sort()

            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i + 1]) / 2

                left_mask = X[col_name] <= split_value
                right_mask = X[col_name] > split_value

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                left_entropy = self.entropy(left_y)
                right_entropy = self.entropy(right_y)

                ig = parent_entropy - (len(left_y) / len(y) * left_entropy + len(right_y) / len(y) * right_entropy)

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

        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1 or self.max_leafs - self.leafs_cnt <= 1) and is_root == False:
            node['leaf'] = y.value_counts(normalize=True).to_dict()
            self.leafs_cnt += 1
            return

        col_name, split_value, ig = self.get_best_split(X, y)

        node[col_name] = {}
        node[col_name][f'<= {split_value}'] = {}
        node[col_name][f'> {split_value}'] = {}

        left_mask = X[col_name] <= split_value
        right_mask = X[col_name] > split_value

        self.fit(X[left_mask], y[left_mask], depth + 1, node[col_name][f'<= {split_value}'])
        self.fit(X[right_mask], y[right_mask], depth + 1, node[col_name][f'> {split_value}'])

    def visualize_tree(self, node=None, graph=None, parent=None, edge_label=''):
        if graph is None:
            graph = Digraph()
        
        if node is None:
            node = self.tree

        for key, value in node.items():
            if key == 'leaf':
                leaf_str = ', '.join([f'{k}: {v:.2f}' for k, v in value.items()])
                graph.node(str(id(node)), f'Leaf: {leaf_str}', shape='box', color='lightblue')
                if parent is not None:
                    graph.edge(parent, str(id(node)), label=edge_label)
            else:
                split_value = list(value.keys())[0].split()[1]
                graph.node(str(id(value)), f'{key} <= {split_value}')
                if parent is not None:
                    graph.edge(parent, str(id(value)), label=edge_label)

                self.visualize_tree(value[f'<= {split_value}'], graph, str(id(value)), '<=' + split_value)
                self.visualize_tree(value[f'> {split_value}'], graph, str(id(value)), '>' + split_value)

        return graph
    
    
    def _class(self, row):
        node = self.tree
        while 'leaf' not in node:
            col_name = list(node.keys())[0]
            split_value = list(node[col_name].keys())[0].split()[1]
            if row[col_name] <= float(split_value):
                node = node[col_name][f'<= {split_value}']
            else:
                node = node[col_name][f'> {split_value}']
        return node['leaf'].get(1, 0)  # Вернуть вероятность класса 1

    def predict_proba(self, X):
        return X.apply(self._class, axis=1)

    def predict(self, X):
        return self.predict_proba(X).apply(lambda x: 1 if x > 0.5 else 0)
    
    
# Создаем и тестируем классификатор
clf = MyTreeClf(max_depth=15, min_samples_split=20, max_leafs=30)
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
