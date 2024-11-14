import numpy as np
import random


X = np.array([[1, 1, 0.3], [1, 0.4, 0.5], [1, 0.7, 0.8]])
y = np.array([1, 1, 0])  
w = np.zeros(3)

act = lambda t: t > 1
predict = lambda x, w: act(np.sum(x * w))

def go(X, y, w, step=1):
    for _ in range(step):
        for i in range(len(X)):
            w = w + (y[i] - predict(X[i], w)) * X[i]
        print(w)
    return w

step = 1
trained_weights = go(X, y, w, step=step)
