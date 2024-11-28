from collections import OrderedDict
import numpy as np
import pandas as pd
from graphviz import Digraph

from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs - 1
        self.leafs_cnt = 0
        self.reserved_leaves = 0
        
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
    
    def fit(self, X, y, depth=0, node=None):
        if node is None:
            self.tree = {}
            node = self.tree
            is_root = True
        else:
            is_root = False

        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1 or 
            self.max_leafs - self.leafs_cnt <= 1) and not is_root:
            node['leaf'] = np.mean(y)
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
                # Отображаем среднее значение в листьях
                graph.node(str(id(node)), f'Leaf: {value:.2f}', shape='box', style='filled', fillcolor='lightgreen')  # Лист зеленый
                if parent is not None:
                    edge_color = 'blue' if '<=' in edge_label else 'red'
                    graph.edge(parent, str(id(node)), label=edge_label, color=edge_color)  # Стрелка с цветом
            else:
                split_value = list(value.keys())[0].split()[1]
                left_node = value[f'<= {split_value}']
                right_node = value[f'> {split_value}']

                # Устанавливаем цвет узла в зависимости от средних значений
                left_leaf_value = left_node.get('leaf', 0)
                right_leaf_value = right_node.get('leaf', 0)

                if parent is None:
                    fill_color = '#FFA500'  # Оранжевый для корня
                else:
                    fill_color = '#FFCCCB' if left_leaf_value > right_leaf_value else '#ADD8E6'  # Красный или голубой для ветвей

                graph.node(str(id(value)), f'{key} <= {split_value}', style='filled', fillcolor=fill_color)

                if parent is not None:
                    edge_color = 'blue' if '<=' in edge_label else 'red'
                    graph.edge(parent, str(id(value)), label=edge_label, color=edge_color)

                self.visualize_tree(left_node, graph, str(id(value)), '<=' + split_value)
                self.visualize_tree(right_node, graph, str(id(value)), '>' + split_value)

        return graph
    
    def sum_leaves(self, node=None):
        if node is None:
            node = self.tree

        total_sum = 0

        for key, value in node.items():
            if key == 'leaf':
                total_sum += value  # Суммируем значение в листе
            else:
                total_sum += self.sum_leaves(value[f'<= {list(value.keys())[0].split()[1]}'])
                total_sum += self.sum_leaves(value[f'> {list(value.keys())[0].split()[1]}'])

        return total_sum
        
# Использование
tree_configs = OrderedDict([
    (1, MyTreeReg(max_depth=3, min_samples_split=2, max_leafs=1)),
    (2, MyTreeReg(max_depth=1, min_samples_split=1, max_leafs=2)),
    (3, MyTreeReg(max_depth=3, min_samples_split=2, max_leafs=5)),
    (4, MyTreeReg(max_depth=5, min_samples_split=100, max_leafs=10)),
    (5, MyTreeReg(max_depth=4, min_samples_split=50, max_leafs=17)),
    (6, MyTreeReg(max_depth=10, min_samples_split=40, max_leafs=21)),
    (7, MyTreeReg(max_depth=15, min_samples_split=35, max_leafs=30)),
])

clf = tree_configs[6]
clf.fit(X, y)

print(clf.leafs_cnt)

# Визуализируем дерево
tree_graph = clf.visualize_tree()
tree_graph.render('decision_tree_reg', format='png', cleanup=True)  # Сохраняем в формате PNG

print(clf.sum_leaves())