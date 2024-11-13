import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


class MyLogReg:
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

        
    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"


    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Добавляем столбец единиц
        self.weights = np.ones(X.shape[1])  # Инициализируем weights с правильной размерностью
        
        for i in range(1, self.n_iter + 1):
            batch_X, batch_y = self._get_mini_batch(X, y)
            y_pred, loss, gradient = self._data_calculations(batch_X, batch_y)
            learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            self.weights -= learning_rate * gradient
            metric_value = self._calculate_metric(batch_y, y_pred)
            
            if verbose and i % verbose == 0:
                print(f"{i} | loss: {loss} | {self.metric}: {metric_value}")
                
        self.best_score = self._calculate_metric(y, self.predict_proba(X))
        

    def y_predict(self, X: np.ndarray):
        z = np.dot(X, self.weights)
        return 1 / (1 + np.exp(-z))

    def get_coef(self):
        return self.weights[1:]  # Возвращаем веса без свободного члена
    

    def _log_loss(self, y_true, y_pred):
        eps = 1e-15
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    
    
    def predict_proba(self, X: np.ndarray):
        return self.y_predict(X)
    
    
    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    
    def _data_calculations(self, X, y):
        regularization, regularization_gradient = self._calculate_regularization()
        y_pred = self.y_predict(X)
        loss = self._log_loss(y, y_pred) + regularization
        gradient = np.dot(X.T, (y_pred - y)) / X.shape[0] + regularization_gradient
        return y_pred, loss, gradient
    
    
    def _calculate_metric(self, y_true, y_pred):
        y_true, y_pred_binary = self._binary_labels(y_true, y_pred)

        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred_binary)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred_binary)
        elif self.metric == 'roc_auc':
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_pred)
            else:
                roc_auc = None
            return roc_auc
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred_binary)
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred_binary)
        else:
            return None


    def _binary_labels(self, y_true, y_pred):
        median = np.median(y_true)
        y_true = (y_true >= median).astype(int)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return y_true, y_pred_binary


    def get_best_score(self):
        return self.best_score
    
    
    def _calculate_regularization(self):
        reg_type = self.reg
        weights = self.weights
        l1_coef = self.l1_coef
        l2_coef = self.l2_coef
        
        if reg_type == 'l1':
            regularization = l1_coef * np.sum(np.abs(weights))
            gradient_reg = l1_coef * np.sign(weights)
        elif reg_type == 'l2':
            regularization = l2_coef * np.sum(weights ** 2)
            gradient_reg = 2 * l2_coef * weights
        elif reg_type == 'elasticnet':
            regularization = (l1_coef * np.sum(np.abs(weights)) +
                             l2_coef * np.sum(weights ** 2))
            gradient_reg = (l1_coef * np.sign(weights) +
                           2 * l2_coef * weights)
        else:
            regularization = 0
            gradient_reg = 0
            
        return regularization, gradient_reg
    
    
    def _get_mini_batch(self, X, y):
        """Returns a mini-batch of data."""
        if self.sgd_sample is not None:
            sample_size = (
                self.sgd_sample if isinstance(self.sgd_sample, int) else
                max(1, int(self.sgd_sample * X.shape[0]))  # sample size as fraction of X.shape[0]
            )
            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            X_batch = X[sample_rows_idx]
            y_batch = y.iloc[sample_rows_idx].values
        else:
            X_batch = X
            y_batch = y.values
        return X_batch, y_batch
        



X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
ln = MyLogReg(n_iter=400, learning_rate=lambda iter: 0.5 * (0.85 ** iter), metric='roc_auc', sgd_sample=0.1, random_state=42)
ln.fit(X, y, verbose=10)
lst = ln.get_coef()
print(f'Среднее значение {sum(lst)/len(lst)}')
print(ln.get_best_score())
