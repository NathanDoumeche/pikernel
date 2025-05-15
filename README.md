# Pikernel

**Pikernel** is a Python package for constructing **physics-informed kernels** as introduced in the paper  
[**Physics-Informed Kernel Learning**](https://arxiv.org/pdf/2409.13786) (2025) by Nathan Doumèche, Francis Bach, Claire Boyer,  and Gérard Biau.

It provides an easy-to-use framework for implementing physics-informed kernels in **1D and 2D**, for a wide class of **ODEs and PDEs with constant coefficients**.  
The package supports both **CPU and GPU** execution and automatically leverages available hardware for optimal performance.



##  Features

- Build kernels tailored to your differential equation constraints  
- Works with any linear ODE/PDE with constant coefficients in 1D or 2D  
- Compatible with NumPy and PyTorch backends  
- GPU support via PyTorch for accelerated computation  



## Installation

You can install the package via pip:

```bash
pip install pikernel
```

## Resources

* **Tutorial:** [https://github.com/claireBoyer/tutorial-piml](https://github.com/claireBoyer/tutorial-piml)
* **Source code:** [https://github.com/NathanDoumeche/pikernel](https://github.com/NathanDoumeche/pikernel)
* **Bug reports:** [https://github.com/NathanDoumeche/pikernel/issues](https://github.com/NathanDoumeche/pikernel/issues)



## Citation
To cite this package:

    @article{doumèche2024physicsinformedkernellearning,
      title={Physics-informed kernel learning},
      author={Nathan Doumèche and Francis Bach and Gérard Biau and Claire Boyer},
      journal={arXiv:2409.13786},
      year={2024}
    }

## Minimal example in 1 dimension 

In this minimal example, the goal is to learn a function $f^\star$ such that $Y = f^\star(X)+\varepsilon$, where
* $Y$ is the target random variable, taking values in $\mathbb R$,
* $X$ is the feature random variable, following the uniform distribution $[-L,L]$ with $L = \pi$,
* $\varepsilon$ is a gaussian noise of distribution $\mathcal N(0, \sigma^2)$, with $\sigma > 0$,
* $f^\star$ is assumed to be $s$ times differentiable, for $s = 2$,
* $f^\star$ is assumed to satisfy the ODE $f'' + f' + f = 0$. 

To this aim, we train a physics-informed kernel on $n = 10^3$ i.i.d. samples $(X_1, Y_1), \dots, (X_n, Y_n)$. This kernel method minimizes the empirical risk
$$L(f) = \frac{1}{n}\sum_{j=1}^n |f(X_i)-Y_i|^2 + \lambda_n \|f\|_{H^s}^2+ \mu_n \int_{[-L,L]} (f''(x)+f'(x)+f(x))^2dx,$$
over the class of function 
$H_m = \{(x\mapsto f(x) = \sum_{k=-m}^m \theta_k \exp(i \pi/L x)), \quad \theta_k \in \mathbb C\}$
where 
* $\lambda_n, \mu_n \geq 0$ are hyperparameters set by the user.
* $ \|f\|_{H^s}$ is the Sobolev norm of order $s$ of $f$.
* the method is discretized over $m = 10^2$ Fourier modes. The higher the number of Fourier modes, the better the approximation capabilities of the kernel. 

Then, we evaluate the kernel on a testing dataset of $l = 10^3$ samples and we compute its RMSE.

The *device* variable from *pikernel.utils* automatically detects whether or not a GPU is available, and run the code on the best hardware available.


```python
import numpy as np
import pandas as pd
import torch

from pikernel.dimension_1 import DifferentialOperator1d, RFF_fit_1d, RFF_estimate_1d
from pikernel.utils import device

# Define the differential operator d/dx
dX = DifferentialOperator1d({(1): 1})

# Define the ODE: f'' + f' + f = 0
PDE = dX**2 + dX + 1

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Parameters
sigma = 0.5       # Noise standard deviation
s = 2             # Smoothness of the solution 
L = np.pi         # Domain: [-L, L]
n = 10**3         # Number of training samples
m = 10**2         # Number of Fourier features
l = 10**3         # Number of test points

# Generate the training data
x_train = torch.rand(n, device=device) * 2 * L - L
y_train = torch.exp(-x_train / 2) * torch.cos(np.sqrt(3) / 2 * x_train) + sigma * torch.randn(n, device=device)

# Generate the test data
x_test = torch.rand(l, device=device) * 2 * L - L
y_test = torch.exp(-x_test / 2) * torch.cos(np.sqrt(3) / 2 * x_test)

# Regularization parameters
lambda_n = np.log(n) / n    # Smoothness hyperparameter
mu_n = 10**4                # PDE hyperparameter

# Fit model using the ODE constraint
regression_vector = RFF_fit_1d(x_train, y_train, s, m, lambda_n, mu_n, L, PDE, device)

# Predict on test data
y_pred = RFF_estimate_1d(regression_vector, x_test, s, m, n, lambda_n, mu_n, L, PDE, device)

# Compute RMSE
rmse = torch.mean((torch.real(y_pred) - y_test) ** 2).item()
print(f"MSE = {rmse}")

```

Output
```bash
RMSE = 0.05328356702649015.
```