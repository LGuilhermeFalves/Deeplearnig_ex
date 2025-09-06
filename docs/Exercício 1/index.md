## Exercício 1 — Separabilidade de classes

### Análise da Distribuição e Sobreposição das Classes

Foi criado o conjunto de dados sintético com 4 classes seguindo os seguintes parâmetros:

- **Classe 0** (média [2,3], desvio [0.8,2.5]): Distribuída próxima ao canto inferior esquerdo, com dispersão maior no eixo y.
- **Classe 1** (média [5,6], desvio [1.2,1.9]): Localizada mais ao centro.
- **Classe 2** (média [8,1], desvio [0.9,0.9]): Mais isolada.
- **Classe 3** (média [15,4], desvio [0.5,2.0]): Bastante afastada das outras classes, com dispersão maior no eixo y.

![alt text](image.png)


### Separação Linear

Ao vizualizar o gráfico é notável que não é possível separar todas as classes com limites lineares simples, apesar da classe 3 estar afastada, há sobreposição de dados entre as classes 0 e 1 oque impossibilitaria a separação.

### Limites de Decisão (Decision Boundaries)

Baseado no gráfico, os limites de decisão que uma rede neural treinada poderia aprender seriam:

- Limites curvos ou não-lineares entre as classes 0 e 1, devido à sobreposição.
- Limites mais simples entre as classes 2 e 3.

#### Código com gração de dados:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros das classes
params = [
	{'mean': [2, 3], 'std': [0.8, 2.5]},
	{'mean': [5, 6], 'std': [1.2, 1.9]},
	{'mean': [8, 1], 'std': [0.9, 0.9]},
	{'mean': [15, 4], 'std': [0.5, 2.0]}
]

num_classes = 4
samples_per_class = 100

X = []
y = []

for i, p in enumerate(params):
	x_class = np.random.normal(loc=p['mean'], scale=p['std'], size=(samples_per_class, 2))
	X.append(x_class)
	y.append(np.full(samples_per_class, i))

X = np.vstack(X)
y = np.concatenate(y)

colors = ['red', 'blue', 'green', 'purple']
labels = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3']

plt.figure(figsize=(8, 6))
for i in range(num_classes):
	plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=labels[i], alpha=0.7)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Distribuição das classes sintéticas (Exercício 1)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico com limites de decisão ilustrativos
plt.figure(figsize=(8, 6))
for i in range(num_classes):
	plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=labels[i], alpha=0.7)

# Limites de decisão aproximados (apenas ilustrativos)
# Entre classe 0 e 1 (reta)
plt.plot([3.8, 4.2], [12.5, -3], 'k--', label='Limite 0/1')
# Entre classe 1 e 2 (reta)
plt.plot([4.1, 12.5], [2.8, 5.2], 'k-.', label='Limite 1/2')
# Entre classe 2 e 3 (reta)
plt.plot([11.30, 13.6], [11.1, -2], 'k:', label='Limite 2/3')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Distribuição das classes com limites de decisão (Exercício 1)')
plt.legend()
plt.grid(True)
plt.show()
