import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None 
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.loss_history = []
        
        
    def __str__(self) -> str:
        return (
            f"MyLineReg(n_iter={self.n_iter}, learning_rate={self.learning_rate}, "
            f"metric={self.metric}, reg={self.reg}, l1_coef={self.l1_coef}, "
            f"l2_coef={self.l2_coef}, sgd_sample={self.sgd_sample}, "
            f"random_state={self.random_state})"
        )

    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = 0) -> None:
        random.seed(self.random_state)
        X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.ones(X_with_intercept.shape[1])
        self.y = y
        self.X = X_with_intercept
        
        for iteration in range(1, self.n_iter + 1):
            batch_X, batch_y = self._get_mini_batch(X_with_intercept, y)
            predictions = np.dot(batch_X, self.weights)
            errors = predictions - batch_y
            regularization, regularization_gradient = self._calculate_reg_and_grad()
            loss = np.mean(errors ** 2) + regularization
            self.loss_history.append(loss)
            gradient = (2 / batch_X.shape[0] * np.dot(batch_X.T, errors)) + regularization_gradient
            
            
            learning_rate = self.learning_rate(iteration) if callable(self.learning_rate) else self.learning_rate
            self.weights -= learning_rate * gradient
            
            if self.metric and iteration % verbose == 0:
                metric_value = self._calculate_metric(y, np.dot(X_with_intercept,    self.weights))
                print(f"Iteration {iteration} | Loss: {loss:.5f} | {self.metric}: {metric_value:.5f}")
        
        self.best_score = self._calculate_metric(y, np.dot(X_with_intercept, self.weights)) if self.metric else None


    def get_coef(self):
        return np.array(self.weights[1:])
    
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        features_with_intercept = np.hstack((np.ones((features.shape[0], 1)), features))
        return np.dot(features_with_intercept, self.weights)
    
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = {
            'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
            'rmse': lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mape': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }

        return metric.get(self.metric, lambda y_true, y_pred: None)(y_true, y_pred)
        
    
    
    def get_best_score(self):
        return self.best_score
    
    
    def _get_mini_batch(self, X, y):
        if self.sgd_sample is not None:
            sample_size = (
                self.sgd_sample if isinstance(self.sgd_sample, int) else
                max(1, int(self.sgd_sample * X.shape[0]))
            )
            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            X_batch = X[sample_rows_idx]
            y_batch = y.iloc[sample_rows_idx].values
        else:
            X_batch = X
            y_batch = y.values
        return X_batch, y_batch
    
    
    def _calculate_reg_and_grad(self):
        regularization = self._calculate_regularization()
        gradient_reg = self._calculate_regularization_gradient()
        return regularization, gradient_reg
    
    
    def _calculate_regularization(self):
        reg = {
            'l1': lambda weights: self.l1_coef * np.sum(np.abs(weights)),
            'l2': lambda weights: self.l2_coef * np.sum(weights ** 2),
            'elasticnet': lambda weights: (self.l1_coef * np.sum(np.abs(weights)) +
                                          self.l2_coef * np.sum(weights ** 2))
        }

        return reg.get(self.reg, lambda weights: 0)(self.weights)
    
    
    def _calculate_regularization_gradient(self):
        reg = {
            'l1': lambda weights: (self.l1_coef * np.sign(weights)),
            'l2': lambda weights: (2 * (self.l2_coef * weights)),
            'elasticnet': lambda weights: ((self.l1_coef * np.sign(weights) +
                                          2 * self.l2_coef * weights))
        }

        return reg.get(self.reg, lambda weights: 0)(self.weights)
    
    
    def plot_loss(self, start=0, step=1):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.n_iter + 1)[start::step], self.loss_history[start::step], marker='o', linestyle='-', color='b')
        plt.title('Изменение Loss по итерациям')
        plt.xlabel('Итерации')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
ln = MyLineReg(n_iter=50, learning_rate=lambda iter: 0.8 * (0.8 ** iter), metric='r2', sgd_sample=0.3, random_state=42, reg='l1', l1_coef=0.000001, l2_coef=0.0001)
ln.fit(X, y, verbose=5)
lst = ln.get_coef()
print(f'Среднее значение параметров: {sum(lst)/len(lst)}')
print(f'Оптимальная метрика {ln.metric}: {ln.get_best_score()}')
ln.plot_loss(2, 2)
