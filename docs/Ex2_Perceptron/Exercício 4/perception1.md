## Exercício 1 — Perception

### Geração de dados

Os dados foram gerados seguindo os seguintes parâmetros

- **Classe 0**  
  - Média: `[1.5, 1.5]`  
  - Matriz de covariância: `[[0.5, 0], [0, 0.5]]`  

- **Classe 1**  
  - Média: `[5, 5]`  
  - Matriz de covariância: `[[0.5, 0], [0, 0.5]]`  

![alt text](<Screenshot from 2025-09-14 20-52-07.png>)

## Implementação do Perceptron

O perceptron foi implementado do zero em Python.  
A regra de atualização utilizada foi:

\[
w = w + n*y*x and b = b + n*y
\]

O treinamento ocorreu até convergência ou até 100 épocas.

```python
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

n_samples = 1000
mean0 = [1.5, 1.5]
cov0 = [[0.5, 0], [0, 0.5]]
mean1 = [5, 5]
cov1 = [[0.5, 0], [0, 0.5]]

# Classe 0
X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
y0 = np.zeros(n_samples)

# Classe 1
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
y1 = np.ones(n_samples)

# Concatena
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# Embaralha
idx = np.arange(len(y))
np.random.shuffle(idx)
X, y = X[idx], y[idx]


class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.accuracies = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Converte labels para -1 e +1
        y_mod = np.where(y == 1, 1, -1)

        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y_mod):
                linear_output = np.dot(xi, self.w) + self.b
                y_pred = 1 if linear_output >= 0 else -1
                if yi != y_pred:
                    # Regra de atualização
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1


            y_pred_all = self.predict(X)
            acc = np.mean(y_pred_all == y)
            self.accuracies.append(acc)

            if errors == 0:  # convergiu
                break

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)

    perc = Perceptron(lr=0.01, epochs=100)
    perc.fit(X, y)

    y_pred = perc.predict(X)
    accuracy = np.mean(y_pred == y)

```

Como as distribuições têm médias distantes e baixa variância, as classes são praticamente linearmente separáveis, fazendo o perceptron convergir rápido

![alt text](<Screenshot from 2025-09-14 21-00-52.png>)