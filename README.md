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

# Minimal examples
## Example in dimension 1

**Setting.**
In this example, the goal is to learn a function $f^\star$ such that $Y = f^\star(X)+\varepsilon$, where
* $Y$ is the target random variable, taking values in $\mathbb R$,
* $X$ is the feature random variable, following the uniform distribution $[-L,L]$ with $L = \pi$,
* $\varepsilon$ is a gaussian noise of distribution $\mathcal N(0, \sigma^2)$, with $\sigma > 0$,
* $f^\star$ is assumed to be $s$ times differentiable, for $s = 2$,
* $f^\star$ is assumed to satisfy the ODE $f'' + f' + f = 0$. 

**Kernel method.** To this aim, we train a physics-informed kernel on $n = 10^3$ i.i.d. samples $(X_1, Y_1), \dots, (X_n, Y_n)$. This kernel method minimizes the empirical risk
$$L(f) = \frac{1}{n}\sum_{j=1}^n |f(X_i)-Y_i|^2 + \lambda_n |f|^2_s+ \mu_n \int_{-L}^L (f''(x)+f'(x)+f(x))^2dx,$$
over the class of functions 
$H_m$, where
* $H_m$ is space of complex-valued trigonometric polynomials of degree at most $m$, i.e., $H_m$ is the class of functions $f$ such that $f(x) = \sum_{k=-m}^m \theta_k \exp(i  \pi k x/(2L))$ for some Fourier coefficients $\theta_k \in \mathbb C$ 
* $\lambda_n, \mu_n \geq 0$ are hyperparameters set by the user.
* $|f|_s$ is the Sobolev norm of order $s$ of $f$.
* the method is discretized over $m = 10^2$ Fourier modes. The higher the number of Fourier modes, the better the approximation capabilities of the kernel. 

Then, we evaluate the kernel on a testing dataset of $l = 10^3$ samples and we compute its RMSE. In this example, the unknown function is $$f^\star(x) = \exp(-x/2) \cos(x\sqrt{3}/2 ).$$

The *device* variable from *pikernel.utils* automatically detects whether or not a GPU is available, and run the code on the best hardware available.

**Differential operator.** To define the ODE $a_1 f + a_2 \frac{d}{dx}f+ \dots + a_{s+1} \frac{d^s}{dx^s}f = 0$, just set the variable *ODE* to $ODE = a_1 + a_2*dX + \dots + a_{s+1} * dX**s$.


```python
import torch
import numpy as np

from pikernel.utils import device
from pikernel.dimension_1 import RFF_fit_1d, RFF_estimate_1d, dX

# Set a seed for reproducibility of the results
torch.manual_seed(1)

# dX is the differential operator d/dx
# Define the ODE: f'' + f' + f = 0
ODE = 1 + dX+ dX**2 

# Parameters
sigma = 0.5       # Noise standard deviation
s = 2             # Smoothness of the solution 
L = torch.pi         # Domain: [-L, L]
n = 10**3         # Number of training samples
m = 10**2         # Number of Fourier features
l = 10**3         # Number of test points

# Generate the training data
scaling = np.sqrt(3) / 2
x_train = torch.rand(n, device=device) * 2 * L - L
y_train = torch.exp(-x_train / 2) * torch.cos(scaling* x_train) + sigma * torch.randn(n, device=device)

# Generate the test data
x_test = torch.rand(l, device=device) * 2 * L - L
ground_truth = torch.exp(-x_test / 2) * torch.cos(scaling* x_test)

# Regularization parameters
lambda_n = 1 / n    # Smoothness hyperparameter
mu_n = 1                    # PDE hyperparameter

# Fit model using the ODE constraint
regression_vector = RFF_fit_1d(x_train, y_train, s, m, lambda_n, mu_n, L, ODE, device)

# Predict on test data
y_pred = RFF_estimate_1d(regression_vector, x_test, s, m, n, lambda_n, mu_n, L, ODE, device)

# Compute the mean squared error
mse = torch.mean((torch.real(y_pred) - ground_truth) ** 2).item()
print(f"MSE = {mse}")
```

Output
```bash
MSE = 0.0006955136680173575
```


## Example in dimension 2

**Setting.**
In this example, the goal is to learn a function $f^\star$ such that $Z = f^\star(X, Y)+\varepsilon$, where
* $Z$ is the target random variable, taking values in $\mathbb R$,
* $X$ and $Y$ are the feature random variables and $(X,Y)$ takes values in $\Omega \subseteq [-L,L]$, for some domain $\Omega$ and some $L>0$
* $\varepsilon$ is a gaussian noise of distribution $\mathcal N(0, \sigma^2)$, with $\sigma > 0$,
* $f^\star$ is assumed to be $s$ times differentiable, for $s = 2$,
* $f^\star$ is assumed to satisfy the heat equation on $\Omega$, i.e.,  $$\forall x \in \Omega, \quad \frac{\partial}{\partial_x} f -\frac{\partial^2}{\partial_y^2} f = 0.$$ 

**Domain.** In this example the domain is $\Omega = [-L,L]^2$. It is possible to consider different domains, by changing the variable *domain*. The available domains are

* the square $\Omega = [-L,L]^2$, by setting $domain = "square"$,
* the disk $\Omega$ made of all points $(x,y)\in \mathbb R^2$ with $x^2+y^2 \leq L^2$, by setting $domain = "disk"$.

**Kernel method.** To this aim, we train a physics-informed kernel on $n = 10^3$ i.i.d. samples $(X_1, Y_1), \dots, (X_n, Y_n)$. This kernel method minimizes the empirical risk
$$L(f) = \frac{1}{n}\sum_{j=1}^n |f(X_i)-Y_i|^2 + \lambda_n |f|^2_s+ \mu_n \int_{\Omega} (\frac{\partial}{\partial_x} f(x) -\frac{\partial^2}{\partial_y^2} f(x))^2dx,$$
over the class of function $H_m$, where
* $H_m$ is space of complex-valued trigonometric polynomials of degree at most $m$, i.e., $H_m$ is the class of functions $f$ such that $f(x) = \sum_{k_1=-m}^m\sum_{k_2=-m}^m \theta_{k_1, k_2} \exp(i \pi (k_1 x+ k_2 y)/(2L) )$ for some Fourier coefficients $\theta_{k_1, k_2} \in \mathbb C$ 
* $\lambda_n, \mu_n \geq 0$ are hyperparameters set by the user.
* $|f|_s$ is the Sobolev norm of order $s$ of $f$.
* the method is discretized over $m = 10^1$ Fourier modes. The higher the number of Fourier modes, the better the approximation capabilities of the kernel. 

Then, we evaluate the kernel on a testing dataset of $l = 10^3$ samples and we compute its RMSE. In this example, the unknown function is $$f^\star(x,y) = \exp(-x) \cos(y ).$$

The *device* variable from *pikernel.utils* automatically detects whether or not a GPU is available, and run the code on the best hardware available.


**Differential operator.** For example, to define the PDE $a_1 f + a_2 \frac{\partial}{\partial x}f+ a_3 \frac{\partial}{ \partial y}f + a_4 \frac{\partial^2}{\partial x \partial y}f + a_5 \frac{\partial^3}{\partial x^3}f= 0$, just set the variable *PDE* to $PDE = a_1 + a_2 * dX+ a_3 * dY + a_4 * dX*dY + a_5 * dX**3$.

```python
import torch

from pikernel.dimension_2 import RFF_fit, RFF_estimate, dX, dY
from pikernel.utils import device

# Set seed for reproducibility
torch.manual_seed(1)

# Define the PDE corresponding to the heat equation: d/dx - d^2/dy^2
PDE = dX - dY**2

# Parameters
sigma = 0.5         # Noise standard deviation
s = 2               # Smoothness of the solution 
L = torch.pi        # The domain is a subset of [-L, L]^2
domain = "square"   # Domain's shape
m = 10              # Number of Fourier features in each dimension
n = 10**3           # Number of training points
l = 10**3           # Number of testing points
      
# Generate the training data
x_train = torch.rand(n, device=device)*2*L-L
y_train = torch.rand(n, device=device)*2*L-L
z_train = torch.exp(-x_train)*torch.cos(y_train) + sigma * torch.randn(n, device=device)

# Generate the test data
x_test = torch.rand(l, device=device)*2*L-L
y_test = torch.rand(l, device=device)*2*L-L
ground_truth =  torch.exp(-x_test)*torch.cos(y_test) 

# Regularization parameters
lambda_n = 1/n   # Smoothness hyperparameter
mu_n = 1         # PDE hyperparameter

# Fit model using the DPE constraint
regression_vector = RFF_fit(x_train, y_train, z_train, s, m, lambda_n, mu_n, L, domain, PDE, device)
z_pred = RFF_estimate(regression_vector, x_test, y_test, s, m, n, lambda_n, mu_n, L, domain, PDE, device)

# Compute the mean squared error
mse = torch.mean(torch.square(torch.abs(z_pred - ground_truth))).item()
print("MSE = ", mse)
```

Output
```bash
MSE =  0.006954170339062708
```