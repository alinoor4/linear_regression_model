# Linear Regression from Scratch

A hands-on implementation of **univariate and multivariate linear regression** built entirely from scratch using NumPy — no scikit-learn, no black boxes. This project walks through the core mathematics of machine learning: the cost function, gradient computation, and gradient descent optimization, applied to real-world datasets.

---

## Project Overview

This repository contains two notebooks, each demonstrating how a machine learning model learns by iteratively minimizing prediction error through gradient descent.

### Notebook 1 — Univariate Linear Regression (`linear_reg.ipynb`)

Predicts employee salary from years of experience using a single input feature.

**Dataset:** `Salary Data.csv`  
**Feature:** `YearsExperience`  
**Target:** `Salary`

### Notebook 2 — Multivariate Linear Regression (`multi_linear_reg.ipynb`)

Extends the model to multiple input features, predicting student performance from study hours and prior academic scores.

**Dataset:** `Student_Performance.csv`  
**Features:** `Hours Studied`, `Previous Scores`  
**Target:** `Performance Index`

---

## Concepts Implemented

### 1. Linear Model

**Univariate:**
```
f(x) = w * x + b
```

**Multivariate:**
```
f(X) = W · X + b
```

where `W` is the weight vector and `b` is the bias (intercept), both initialized to zero.

### 2. Cost Function (Mean Squared Error)

Measures how far off the model's predictions are from the actual values:

```
J(w, b) = (1 / 2m) * Σ (f(x_i) - y_i)²
```

### 3. Gradient Computation

Computes the partial derivatives of the cost function with respect to `w` (or `W`) and `b`:

**Univariate:**
```
∂J/∂w = (1/m) * Σ (f(x_i) - y_i) * x_i
∂J/∂b = (1/m) * Σ (f(x_i) - y_i)
```

**Multivariate:**
```
∂J/∂W_j = (1/m) * Σ (f(X_i) - y_i) * X_i,j    for each feature j
∂J/∂b   = (1/m) * Σ (f(X_i) - y_i)
```

### 4. Gradient Descent

Iteratively updates the parameters to minimize cost:

```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

| Notebook | Learning Rate | Iterations | Cost Logged Every |
|---|---|---|---|
| `linear_reg.ipynb` | 0.01 | 10,000 | 1,000 steps |
| `multi_linear_reg.ipynb` | 0.01 | 1,000 | 100 steps |

---

## Visualizations

### Univariate (`linear_reg.ipynb`)
- **Scatter plot** of Salary vs. Years of Experience
- **Prediction line overlay** plotted against training data after gradient descent converges

### Multivariate (`multi_linear_reg.ipynb`)
- **Correlation heatmap** (seaborn) across all dataset features
- **Side-by-side scatter plots** of Performance Index vs. Hours Studied and vs. Previous Scores
- **Prediction overlay** plots for each individual feature

---

## Project Structure

```
├── linear_reg.ipynb          # Univariate regression notebook
├── multi_linear_reg.ipynb    # Multivariate regression notebook
├── Salary Data.csv           # Dataset for univariate model
├── Student_Performance.csv   # Dataset for multivariate model
└── README.md
```

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn
```

No other dependencies — the regression algorithms are fully hand-coded.

---

## License

MIT