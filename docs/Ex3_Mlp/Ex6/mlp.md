# Exercício 1 — Mlp

Nessa atividade foi feita a implementação passo a passo (forward + backprop) para uma MLP com 2 entradas, 1 camada oculta com 2 neurônios e 1 saída. A ativação usada é **tanh** e a atualização de parâmetros foi feita com learning rate n = 0.3.

---

## Código (Python)

```python
import numpy as np

# Dados fornecidos
x = np.array([0.5, -0.2], dtype=float)
y = 1.0

W1 = np.array([[0.3, -0.1],
               [0.2,  0.4]], dtype=float)
b1 = np.array([0.1, -0.2], dtype=float)

W2 = np.array([0.5, -0.3], dtype=float)
b2 = 0.2

n = 0.3  # learning rate (ajustado)

def tanh(u):
    return np.tanh(u)

def dtanh(u):
    return 1.0 - np.tanh(u)**2

# Forward pass
u1 = W1.dot(x) + b1
h1 = tanh(u1)
u2 = W2.dot(h1) + b2
y_hat = tanh(u2)
L = 0.5 * (y - y_hat)**2

# Backprop
dL_dyhat = -(y - y_hat)
dyhat_du2 = dtanh(u2)
delta2 = dL_dyhat * dyhat_du2

dL_dW2 = delta2 * h1
dL_db2 = delta2

dL_dh1 = delta2 * W2
dh1_du1 = dtanh(u1)
delta1 = dL_dh1 * dh1_du1

dL_dW1 = np.outer(delta1, x)
dL_db1 = delta1

# Updates (gradient descent) with n = 0.3
W2_new = W2 - n * dL_dW2
b2_new = b2 - n * dL_db2

W1_new = W1 - n * dL_dW1
b1_new = b1 - n * dL_db1

# Print results
np.set_printoptions(precision=6, suppress=True)
print("Forward pass:")
print("u1 =", u1)
print("h1 =", h1)
print("u2 =", u2)
print("y_hat =", y_hat)
print("Loss L =", L)

print("\nGradients:")
print("dL/dW2 =", dL_dW2)
print("dL/db2 =", dL_db2)
print("dL/dW1 =\n", dL_dW1)
print("dL/db1 =", dL_db1)

print("\nParameter updates (n=0.3):")
print("W2_old =", W2)
print("W2_new =", W2_new)
print("b2_old =", b2)
print("b2_new =", b2_new)
print("W1_old =\n", W1)
print("W1_new =\n", W1_new)
print("b1_old =", b1)
print("b1_new =", b1_new)
```

---

## Saídas (resultados numéricos)

**Forward pass:**
```
u1 = [ 0.27 -0.18]
h1 = [ 0.263625 -0.178081]
u2 = 0.38523667817130075
y_hat = 0.36724656264510797
Loss L = 0.2001884562422156
```

**Gradients:**
```
dL/dW2 = [-0.144312  0.097484]
dL/db2 = -0.5474139573567998
dL/dW1 =
 [[-0.127342  0.050937]
 [ 0.079508 -0.031803]]
dL/db1 = [-0.254685  0.159016]
```

**Parameter updates (n=0.3):**
```
W2_old = [ 0.5 -0.3]
W2_new = [ 0.543294 -0.329245]
b2_old = 0.2
b2_new = 0.36422418720704

W1_old =
 [[ 0.3 -0.1]
 [ 0.2  0.4]]
W1_new =
 [[ 0.338203 -0.115281]
 [ 0.176148  0.409541]]

b1_old = [ 0.1 -0.2]
b1_new = [ 0.176405 -0.247705]
```

---

## Breve explicação
- **Forward**: calcula pre-ativação e ativação na camada oculta (`u1`, `h1`), em seguida a pre-ativação e ativação de saída (`u2`, `y_hat`) e a perda MSE.
- **Backprop**: computa `delta2` (gradiente no nó de saída), depois os gradientes de `W2` e `b2`, propaga para a camada oculta (obtendo `delta1`) e calcula gradientes de `W1` e `b1`.
- **Atualização**: aplica gradiente descendente com `n = 0.3` para obter os novos parâmetros.
