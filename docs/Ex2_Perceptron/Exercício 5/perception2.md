# Exercício 2 — Perceptron

## Geração de dados

Foram geradas 1000 amostras por classe,com os seguintes parâmetros:

- **Classe 0**  
  - Média: `[3, 3]`  
  - Matriz de covariância: `[[1.5, 0], [0, 1.5]]`  

- **Classe 1**  
  - Média: `[4, 4]`  
  - Matriz de covariância: `[[1.5, 0], [0, 1.5]]`  

Esses parâmetros criam uma  parcial entre as classes, já que as médias estão próximas e a variância é alta.

![alt text](<Screenshot from 2025-09-14 21-08-50.png>)


---

## Implementação do Perceptron

A implementação foi a mesma do **Exercício 1**, utilizando a regra de atualização do perceptron com taxa de aprendizado n= 0.01, treinando até convergência ou até 100 épocas.

```python
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
        y_mod = np.where(y == 1, 1, -1)

        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y_mod):
                linear_output = np.dot(xi, self.w) + self.b
                y_pred = 1 if linear_output >= 0 else -1
                if yi != y_pred:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1

            y_pred_all = self.predict(X)
            acc = np.mean(y_pred_all == y)
            self.accuracies.append(acc)

            if errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)
```

![alt text](<Screenshot from 2025-09-14 21-12-01.png>)

O perceptron não consegue convergir totalmente porque os dados não são linearmente separáveis. A sobreposição entre as duas distribuições (médias próximas e variância maior) cria pontos que sempre estarão no lado errado da fronteira

![alt text](<Screenshot from 2025-09-14 21-14-02.png>)