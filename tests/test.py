import torch
import numpy as np
from pikernel.utils import find_device
from pikernel.kernel import PikernelModel
from pikernel.dimension_1 import dX, Sob_matrix_1d
from pikernel.dimension_2 import dX_1, dX_2



def test_import():
    assert True

def test_sob_mat_1d():
    m = 1
    s = 1
    L = 1
    
    device = find_device()
    s_mat = Sob_matrix_1d(m, s, L, device)

    assert True

def test_minimal_example_1d():
    # Set a seed for reproducibility of the results
    torch.manual_seed(1)

    # dX is the differential operator d/dx
    # Define the ODE: f'' + f' + f = 0
    dimension = 1
    ODE = 1 + dX+ dX**2 

    # Parameters
    device = find_device() # Automatically detects GPU, or CPU
    sigma = 0.5            # Noise standard deviation
    s = 2                  # Smoothness of the solution 
    L = torch.pi           # Domain: [-L, L]
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
    assert mse


def test_minimal_example_2d():
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
    assert mse