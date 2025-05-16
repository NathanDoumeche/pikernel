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
* $X$ is the feature random variable, taking values in $[-L,L]$,
* $\varepsilon$ is a random noise, i.e., $\mathbb E(\varepsilon \mid X)=0$,
* the distribution of $X$, $Y$, and $\varepsilon$ are unknown for the user,
* $f^\star$ is assumed to be $s$ times differentiable,
* $f^\star$ is assumed to satisfy a known ODE. 

In this example, $L = \pi$, $s = 2$, $\varepsilon$ is a gaussian noise of distribution $\mathcal N(0, \sigma^2)$ with $\sigma = 0.5$. Moreover,  $f^\star$ satisfies the ODE $f'' + f' + f = 0$.

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

**Differential operator.** In the *pikernel* framework, ODEs are stored into a specific *ODE* variable. To define the ODE $a_1 f + a_2 \frac{d}{dx}f+ \dots + a_{s+1} \frac{d^s}{dx^s}f = 0$, just set the variable *ODE* to $ODE = a_1 + a_2*dX + \dots + a_{s+1} * dX**s$. In this specific example, the ODE $f''+f'+f=0$ translates into $ODE = 1 + dX + dX**2$.


```python
import torch
import numpy as np

from pikernel.utils import find_device
from pikernel.kernel import PikernelModel
from pikernel.dimension_1 import dX

# Set a seed for reproducibility of the results
torch.manual_seed(1)

# dX is the differential operator d/dx
# Define the ODE: f'' + f' + f = 0
dimension = 1
ODE = 1 + dX+ dX**2 

# Parameters
device = find_device() # Automatically detects GPU, or CPU
sigma = 0.5            # Noise standard deviation
s = 2                  # A priori smoothness of the solution 
L = torch.pi           # Domain where the ODE holds: [-L, L]
n = 10**3              # Number of training samples
m = 10**2              # Number of Fourier features
l = 10**3              # Number of test points

# Generate the training data
scaling = np.sqrt(3) / 2
x_train = torch.rand(n, device=device) * 2 * L - L
y_train = torch.exp(-x_train / 2) * torch.cos(scaling* x_train) + sigma * torch.randn(n, device=device)

# Generate the test data
x_test = torch.rand(l, device=device) * 2 * L - L
ground_truth = torch.exp(-x_test / 2) * torch.cos(scaling* x_test)

# Regularization parameters
lambda_n = 1 / n    # Smoothness hyperparameter
mu_n = 1            # PDE hyperparameter

# Fit model using the ODE constraint
kernel_model = PikernelModel(dimension, L, ODE, device)
kernel_model.fit(x_train, y_train, s, m, lambda_n, mu_n, n)

# Predict on test data
y_pred = kernel_model.predict(x_test)

# Compute the mean squared error
mse = kernel_model.mse(y_pred, ground_truth)
print(f"MSE = {mse}")
```

Output
```bash
MSE = 0.0006955136680173575
```


## Example in dimension 2

**Setting.**
In this example, the goal is to learn a function $f^\star$ such that $Y = f^\star(X_1, X_2)+\varepsilon$, where
* $Y$ is the target random variable, taking values in $\mathbb R$,
* $X_1$ and $X_2$ are the feature random variables and $(X_1,X_2)$ takes values in $\Omega \subseteq [-L,L]$, for some domain $\Omega$ and some $L>0$
* $\varepsilon$ is a random noise, i.e., $\mathbb E(\varepsilon \mid X_1, X_2)=0$,
* the distribution of $X_1$, $X_2$, and $\varepsilon$ are unknown for the user,
* $f^\star$ is assumed to be $s$ times differentiable, 
* $f^\star$ is assumed to satisfy a known PDE.

In this example, $L = \pi$, $s = 2$, and $\varepsilon$ is a gaussian noise of distribution $\mathcal N(0, \sigma^2)$ with $\sigma = 0.5$. Moreover, $f^\star$ is a solution to the heat equation on $\Omega$, i.e.,  $$\forall x \in \Omega, \quad \frac{\partial}{\partial_1} f -\frac{\partial^2}{\partial_2^2} f = 0.$$ 

**Domain.** In this example the domain is $\Omega = [-L,L]^2$. It is possible to consider different domains, by changing the variable *domain*. The available domains are

* the square $\Omega = [-L,L]^2$, by setting *domain = "square"*,
* the disk $\Omega$ made of all points $(x_1,x_2)\in \mathbb R^2$ with $x_1^2+x_2^2 \leq L^2$, by setting *domain = "disk"*.

**Kernel method.** To this aim, we train a physics-informed kernel on $n = 10^3$ i.i.d. samples $(X_{1,1}, X_{2,1}, Y_1), \dots, (X_{1,n}, X_{2,n}, Y_n)$. This kernel method minimizes the empirical risk
$$L(f) = \frac{1}{n}\sum_{j=1}^n |f(X_{1,i}, X_{2,i})-Y_i|^2 + \lambda_n |f|^2_s+ \mu_n \int_{\Omega} (\frac{\partial}{\partial_1} f(x_1, x_2) -\frac{\partial^2}{\partial_2^2} f(x_1,x_2))^2dx_1dx_2,$$
over the class of function $H_m$, where
* $H_m$ is space of complex-valued trigonometric polynomials of degree at most $m$, i.e., $H_m$ is the class of functions $f$ such that $f(x_1, x_2) = \sum_{k_1=-m}^m\sum_{k_2=-m}^m \theta_{k_1, k_2} \exp(i \pi (k_1 x_1+ k_2 x_2)/(2L) )$ for some Fourier coefficients $\theta_{k_1, k_2} \in \mathbb C$ 
* $\lambda_n, \mu_n \geq 0$ are hyperparameters set by the user.
* $|f|_s$ is the Sobolev norm of order $s$ of $f$.
* the method is discretized over $m = 10^1$ Fourier modes. The higher the number of Fourier modes, the better the approximation capabilities of the kernel. 

Then, we evaluate the kernel on a testing dataset of $l = 10^3$ samples and we compute its RMSE. In this example, the unknown function is $$f^\star(x_1,x_2) = \exp(-x_1) \cos(x_2).$$

The *device* variable from *pikernel.utils* automatically detects whether or not a GPU is available, and run the code on the best hardware available.


**Differential operator.** 
In the *pikernel* framework, PDEs are stored into a specific *PDE* variable. For example, to define the PDE $a_1 f + a_2 \frac{\partial}{\partial 1}f+ a_3 \frac{\partial}{ \partial 2}f + a_4 \frac{\partial^2}{\partial 1 \partial 2}f + a_5 \frac{\partial^3}{\partial 1^3}f= 0$, just set the variable *PDE* to $PDE = a_1 + a_2 * dX_1+ a_3 * dX_2 + a_4 * dX_1*dX_2 + a_5 * dX_1**3$.
 In the following example, the heat equation $\frac{\partial}{\partial_1} f -\frac{\partial^2}{\partial_2^2} f=0$ translates into $PDE = dX_1 - dX_2**2$.

```python
import torch

from pikernel.utils import find_device
from pikernel.kernel import PikernelModel
from pikernel.dimension_2 import dX_1, dX_2

# Set seed for reproducibility
torch.manual_seed(1)

# Define the heat equation PDE: d/dx - d^2/dy^2
dimension = 2
PDE = dX_1 - dX_2**2

# Parameters
device = find_device()   # Automatically detects GPU, or CPU
sigma = 0.5              # Noise standard deviation
s = 2                    # Smoothness of the solution 
L = torch.pi             # The domain is a subset of [-L, L]^2
domain = "square"        # Domain's shape
m = 10                   # Number of Fourier features in each dimension
n = 10**3                # Number of training points
l = 10**3                # Number of testing points
      
# Generate the training data
x1_train = torch.rand(n, device=device)*2*L-L
x2_train = torch.rand(n, device=device)*2*L-L
x_train = [x1_train, x2_train]
y_train = torch.exp(-x1_train)*torch.cos(x2_train) + sigma * torch.randn(n, device=device)

# Generate the test data
x1_test = torch.rand(l, device=device)*2*L-L
x2_test = torch.rand(l, device=device)*2*L-L
x_test = [x1_test, x2_test]
ground_truth =  torch.exp(-x1_test)*torch.cos(x2_test) 

# Regularization parameters
lambda_n = 1/n   # Smoothness hyperparameter
mu_n = 1         # PDE hyperparameter

# Fit model using the PDE constraint
kernel_model = PikernelModel(dimension, L, PDE, device)
kernel_model.fit(x_train, y_train, s, m, lambda_n, mu_n, n)

# Predict on test data
y_pred = kernel_model.predict(x_test)

# Compute the mean squared error
mse = kernel_model.mse(y_pred, ground_truth)
print("MSE = ", mse)
```

Output
```bash
MSE =  0.006954170339062708
```