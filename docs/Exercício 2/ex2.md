## Exercício 2 — Não-linearidade em Dimensões Superiores

### Geração dos Dados

Os dados foram gerados usando distribuições normais multivariadas conforme os seguintes parâmetros:

- **Classe A:** média `[0, 0, 0, 0, 0]`, matriz de covariância.
- **Classe B:** média `[1.5, 1.5, 1.5, 1.5, 1.5]`, matriz de covariância.

### Redução de Dimensionalidade e Visualização

Foi utilizada a técnica de **Análise de Componentes Principais (PCA)** para projetar os dados de 5 dimensões em 2D. O gráfico abaixo mostra a projeção dos dados:

![alt text](<Screenshot from 2025-09-05 21-36-05.png>)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parâmetros Classe A
mean_A = [0, 0, 0, 0, 0]
cov_A = [
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0]
]
mean_B = [1.5, 1.5, 1.5, 1.5, 1.5]
cov_B = [
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5]
]
n_samples = 500

X_A = np.random.multivariate_normal(mean_A, cov_A, n_samples)
X_B = np.random.multivariate_normal(mean_B, cov_B, n_samples)
X = np.vstack([X_A, X_B])
y = np.array([0]*n_samples + [1]*n_samples)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', label='Classe A', alpha=0.6)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Classe B', alpha=0.6)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Projeção PCA dos dados 5D para 2D')
plt.legend()
plt.grid(True)
plt.show()
```

A projeção PCA mostra que as classes A e B possuem regiões de sobreposição e não são linearmente separáveis no espaço 2D projetado. Isso indica que, mesmo no espaço original 5D, um modelo linear simples teria dificuldade em separar as classes com precisão. Para esse tipo de estrutura de dados, redes neurais profundas com funções de ativação não-lineares são mais adequadas, pois conseguem aprender limites de decisão complexos e não-lineares.