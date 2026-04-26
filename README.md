# Linear Regression from Scratch

A hands-on implementation of **univariate linear regression** built entirely from scratch using NumPy — no scikit-learn, no black boxes. This project walks through the core mathematics of machine learning: the cost function, gradient computation, and gradient descent optimization, applied to a real-world salary prediction dataset.

---

## Project Overview

This notebook demonstrates how a machine learning model learns by iteratively minimizing prediction error. Given a dataset of years of experience and corresponding salaries, the model learns the best-fit line through gradient descent.

**Dataset:** `Salary Data.csv`  
**Features:** `YearsExperience`  
**Target:** `Salary`

---

## Concepts Implemented

### 1. Linear Model
The prediction function takes the form:

```
f(x) = w * x + b
```

where `w` is the weight (slope) and `b` is the bias (intercept), both initialized to zero.

### 2. Cost Function (Mean Squared Error)
Measures how far off the model's predictions are from the actual values:

```
J(w, b) = (1 / 2m) * Σ (f(x_i) - y_i)²
```

### 3. Gradient Computation
Computes the partial derivatives of the cost function with respect to `w` and `b`:

```
∂J/∂w = (1/m) * Σ (f(x_i) - y_i) * x_i
∂J/∂b = (1/m) * Σ (f(x_i) - y_i)
```

### 4. Gradient Descent
Iteratively updates the parameters to minimize cost:

```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

Trained with `learning_rate = 0.01` over `10,000 iterations`, with cost logged every 1,000 steps to monitor convergence.

---

## Visualizations

- **Scatter plot** of Salary vs. Years of Experience
- **Prediction line overlay** plotted against training data after gradient descent converges

---

## Project Structure

```
├── basic.ipynb          # Main notebook
├── Salary Data.csv      # Training dataset
└── README.md
```

---

## Requirements

```bash
pip install numpy pandas matplotlib
```

No other dependencies — the regression algorithm is fully hand-coded.

---
## License

MIT
