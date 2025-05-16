from pikernel.dimension_1 import RFF_estimate_1d, RFF_fit_1d
from pikernel.dimension_2 import RFF_estimate, RFF_fit
from pikernel.utils import torch


import torch


class PikernelModel():
    """
    Physics-Informed Kernel Regression Model using Fourier Features.

    This class provides an interface for fitting and predicting solutions to PDEs or ODEs
    using physics-informed machine learning techniques based on kernel approximations.
    
    Attributes:
        dimension (int): Dimensionality of the problem (1 for ODE, 2 for PDE).
        L (float): Domain size.
        PDE (callable): Function representing the PDE/ODE operator.
        device (torch.device): Computation device (CPU or CUDA).
        m (int): Number of Fourier features for the kernel approximation.
        lambda_n (float): Regularization parameter related to kernel smoothness.
        mu_n (float): Regularization parameter related to physical constraints.
        n (int): Number of training points.
        domain (str): Type of spatial domain ('square' by default). Not needed in dimension 1.
    """

    def __init__(self, dimension, L, PDE, device, domain = "square"):
        """
        Initializes the PikernelModel.

        Args:
            dimension (int): 1 for ODEs, 2 for PDEs.
            L (float): Domain size.
            PDE (callable): Differential operator (ODE or PDE) as a function.
            device (torch.device): Torch device to perform computations.
            m (int): Number of Fourier features.
            lambda_n (float): Regularization parameter for kernel norm.
            mu_n (float): Regularization parameter for enforcing physics.
            n (int): Number of training points.
            domain (str, optional): Domain type ('square' or others). Defaults to "square". Not needed in dimension 1.
        """
        self.dimension = dimension
        self.L = L
        self.PDE = PDE
        self.device = device
        self.domain = domain

    def fit(self, x_train, y_train, s, m, lambda_n, mu_n, n):
        """
            Fits the model to training data using a physics-informed kernel regression method.

            Args:
                x_train (Tensor or tuple of Tensors): Training input locations.
                    - If dimension == 1: Tensor of shape (n_samples,).
                    - If dimension == 2: Tuple (x1_train, x2_train), each of shape (n_samples,).
                y_train (Tensor): Observed outputs corresponding to x_train.

            Returns:
                Tensor: Learned regression coefficients.
        """
        self.m = m
        self.lambda_n = lambda_n
        self.mu_n = mu_n
        self.n = n
        self.s = s
        if self.dimension == 1:
            self.regression_vector = RFF_fit_1d(x_train, y_train, self.s, self.m, self.lambda_n, self.mu_n, self.L, self.PDE, self.device)
        else:
            x1_train = x_train[0]
            x2_train = x_train[1]
            self.regression_vector = RFF_fit(x1_train, x2_train, y_train, self.s, self.m, self.lambda_n, self.mu_n, self.L, self.domain, self.PDE, self.device)
        return self.regression_vector

    def predict(self, x_test):
        """
        Predicts the output at new test points using the fitted model.

        Args:
            x_test (Tensor or tuple of Tensors): Test input locations.
                - If dimension == 1: Tensor of shape (n_test,).
                - If dimension == 2: Tuple (x1_test, x2_test), each of shape (n_test,).

        Returns:
            Tensor: Predicted values at the test locations.
        """
        if self.dimension == 1:
            y_pred = RFF_estimate_1d(self.regression_vector, x_test, self.s, self.m, self.n, self.lambda_n, self.mu_n, self.L, self.PDE, self.device)
        else:
            x1_test = x_test[0]
            x2_test = x_test[1]
            y_pred = RFF_estimate(self.regression_vector, x1_test, x2_test, self.s, self.m,self.n, self.lambda_n, self.mu_n, self.L, self.domain, self.PDE, self.device)
        return y_pred

    def mse(self, y_pred, ground_truth):
       """
        Computes the mean squared error between the predictions and the ground truth.

        Args:
            y_pred (Tensor): Predicted outputs.
            ground_truth (Tensor): True values.

        Returns:
            float: Mean squared error.
        """
       mse = torch.mean((torch.real(y_pred) - ground_truth) ** 2).item()
       return mse