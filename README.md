## Implementation of Logistic Regression Using Gradient Descent

### AIM

To write and verify a Python program to implement Logistic Regression from scratch using Gradient Descent for binary classification.

### Equipments Required

* **Hardware:** PC / Laptop
* **Software:** Anaconda / Python 3.7+ Installation, Jupyter Notebook (with `numpy` and `matplotlib`)

---

### Algorithm

1. Prepare the feature matrix $X$ (with $m$ samples and $n$ features) and binary target vector $y \in \{0, 1\}$.
2. Standardize features so gradient descent converges smoothly and quickly, and initialize weights $w = \mathbf{0}$, bias $b = 0$, learning rate $\alpha$, and number of iterations.
3. Compute the linear logit $z = Xw + b$ and pass it through the sigmoid activation function to obtain predicted probabilities $\hat{y} \in (0, 1)$:

$$\hat{y}^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}$$


4. Compute the binary cross-entropy (log-loss) cost function $J(w, b)$:

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$


5. Compute the gradients of the cost function with respect to $w$ and $b$:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$


6. Update the parameters simultaneously:

$$w := w - \alpha \frac{\partial J}{\partial w}, \qquad b := b - \alpha \frac{\partial J}{\partial b}$$


7. Repeat Steps 3–6 for the set number of iterations, then classify samples using a threshold (default $0.5$ on $\hat{y}$) and plot the cost curve and decision boundary.

---

### Program

```python
"""
Program to implement the Logistic Regression Using Gradient Descent.
Developed by: Darshan J
RegisterNumber: 212224020008
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_cost(y, y_hat):
    m = len(y)
    eps = 1e-15
    y_hat_safe = np.clip(y_hat, eps, 1.0 - eps)
    cost = -(1.0 / m) * np.sum(y * np.log(y_hat_safe) + (1.0 - y) * np.log(1.0 - y_hat_safe))
    return cost


def fit_logistic_regression_gd(X, y, lr=0.1, epochs=1500):
    m, n = X.shape
    w = np.zeros(n, dtype=float)
    b = 0.0
    cost_history = []

    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        error = y_hat - y
        dw = (1.0 / m) * np.dot(X.T, error)
        db = (1.0 / m) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        cost_history.append(compute_cost(y, y_hat))

    return w, b, cost_history


def predict_proba(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)


# 1. Sample Dataset: [Study_Hours, Attendance_Percentage] -> Pass (1) / Fail (0)
X = np.array([
    [1.0, 50.0],
    [1.5, 55.0],
    [2.0, 52.0],
    [2.5, 60.0],
    [3.0, 58.0],
    [3.5, 65.0],
    [4.0, 70.0],
    [4.5, 68.0],
    [5.0, 75.0],
    [5.5, 80.0],
    [6.0, 78.0],
    [6.5, 85.0],
    [7.0, 88.0],
    [7.5, 90.0],
    [8.0, 92.0],
    [8.5, 95.0]
], dtype=float)

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)

# 2. Standardize features for smooth gradient descent
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_scaled = (X - mean) / std

# 3. Train model using Gradient Descent
learning_rate = 0.1
num_epochs = 1500
w, b, cost_hist = fit_logistic_regression_gd(X_scaled, y, lr=learning_rate, epochs=num_epochs)

# 4. Evaluate on training data and predict for a new student (4.8 hrs, 74% attendance)
preds = predict(X_scaled, w, b)
accuracy = np.mean(preds == y) * 100

new_sample = np.array([[4.8, 74.0]])
new_sample_scaled = (new_sample - mean) / std
new_prob = predict_proba(new_sample_scaled, w, b)[0]
new_class = predict(new_sample_scaled, w, b)[0]

print(f"Learned Weights (w): {w}")
print(f"Learned Bias (b): {b:.4f}")
print(f"Final Log-Loss Cost: {cost_hist[-1]:.4f}")
print(f"Training Accuracy: {accuracy:.2f}%")
print(f"New Sample [4.8 hrs, 74%] -> Probability: {new_prob:.4f}, Predicted Class: {new_class}")

# 5. Plot Cost Convergence and Decision Boundary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(range(1, num_epochs + 1), cost_hist, color="purple", linewidth=2)
ax1.set_title("Log-Loss Cost vs. Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary Cross-Entropy Cost")
ax1.grid(True)

ax2.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="x", s=70, label="Class 0")
ax2.scatter(X[y == 1, 0], X[y == 1, 1], color="green", marker="o", s=70, label="Class 1")

# Decision boundary in original scale: w0*x0_s + w1*x1_s + b = 0
x0_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
x0_s_vals = (x0_vals - mean[0]) / std[0]
if abs(w[1]) > 1e-8:
    x1_s_vals = -(w[0] * x0_s_vals + b) / w[1]
    x1_vals = x1_s_vals * std[1] + mean[1]
    ax2.plot(x0_vals, x1_vals, color="blue", linewidth=2, label="Decision Boundary")

ax2.set_title("Logistic Regression Decision Boundary")
ax2.set_xlabel("Study Hours")
ax2.set_ylabel("Attendance (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

```

---

### Output

```text
Learned Weights (w): [2.5016 2.1112]
Learned Bias (b): 0.8709
Final Log-Loss Cost: 0.1456
Training Accuracy: 93.75%
New Sample [4.8 hrs, 74%] -> Probability: 0.8690, Predicted Class: 1

```

---

### Result

Thus, the program to implement Logistic Regression using Gradient Descent was written and verified using Python programming.
