import torch
import matplotlib.pyplot as plt
import numpy as np

class Perceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Данные
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Модель
model = Perceptron()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Обучение
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()

# Вывод весов
weights = model.linear.weight.detach().numpy().flatten()
bias = model.linear.bias.detach().numpy()

# Печать весов
print(*weights, *bias)

# Создание графика
plt.figure(figsize=(8, 6))

# Отображение точек данных
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='bwr', s=100, edgecolor='k', label='Data points')

# Построение разделяющей линии
x_line = np.linspace(-0.5, 1.5, 100)
y_line = -(weights[0] * x_line + bias) / weights[1]

plt.plot(x_line, y_line, color='green', label='Decision boundary')

# Настройки графика
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Decision Boundary')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
