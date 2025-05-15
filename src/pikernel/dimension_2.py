import torch
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from pikernel.utils import *

def Sob_formula(k1, k2, j1, j2, s, L):
    return torch.where(torch.logical_and(k1 == j1, k2 == j2), 1+ (k1**2 + k2**2)**s/(2*L)**(2*s), 0.)


def Sob_matrix(m, s, L, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
  k1 = k1.flatten()
  k2 = k2.flatten()
  j1 = j1.flatten()
  j2 = j2.flatten()

  sob_values = Sob_formula(k1, k2, j1, j2, s, L)

  return sob_values.view((2*m+1)**2, (2*m+1)**2)


class DifferentialOperator:
    def __init__(self, coefficients=None):
        """
        Initialize the PDE.
        The keys are tuples representing the powers of d/dX and d/dY respectively.
        For example, {(2, 1): 3, (0, 0): -1} represents 3d^2/dX^2 d/dY - 1.
        """
        if coefficients is None:
            self.coefficients = {}
        else:
            self.coefficients = coefficients

    def __repr__(self):
        terms = []
        for (x_power, y_power), coefficient in sorted(self.coefficients.items(), reverse=True):
            if coefficient == 0:
                continue
            term = f"{coefficient}"
            if x_power != 0:
                term += f"*(d/dX)^{x_power}"
            if y_power != 0:
                term += f"*(d/dY)^{y_power}"
            terms.append(term)
        PDE = " + ".join(terms) if terms else "0"
        return "The PDE of your model is " + PDE + " = 0."

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = DifferentialOperator(self.coefficients.copy())
            if (0, 0) in result.coefficients:
                result.coefficients[(0, 0)] += other
            else:
                result.coefficients[(0, 0)] = other
            return result

        result = DifferentialOperator(self.coefficients.copy())
        for (x_power, y_power), coefficient in other.coefficients.items():
            if (x_power, y_power) in result.coefficients:
                result.coefficients[(x_power, y_power)] += coefficient
            else:
                result.coefficients[(x_power, y_power)] = coefficient
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        result = DifferentialOperator(self.coefficients.copy())
        for (x_power, y_power), coefficient in other.coefficients.items():
            if (x_power, y_power) in result.coefficients:
                result.coefficients[(x_power, y_power)] -= coefficient
            else:
                result.coefficients[(x_power, y_power)] = -coefficient
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = DifferentialOperator()
            for (x_power, y_power), coefficient in self.coefficients.items():
                result.coefficients[(x_power, y_power)] = coefficient * other
            return result

        result = DifferentialOperator()
        for (x1, y1), c1 in self.coefficients.items():
            for (x2, y2), c2 in other.coefficients.items():
                power = (x1 + x2, y1 + y2)
                if power in result.coefficients:
                    result.coefficients[power] += c1 * c2
                else:
                    result.coefficients[power] = c1 * c2
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        if exponent == 0:
            return DifferentialOperator({(0, 0): 1})
        elif exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")

        result = DifferentialOperator(self.coefficients.copy())
        for _ in range(1, exponent):
            result *= self
        return result

    def evaluate(self, x, y, L):
        total = 0
        geometry = -1j*torch.pi/2/L
        for (x_power, y_power), coefficient in self.coefficients.items():
            total += coefficient * (x ** x_power) * (y ** y_power) * (geometry **(x_power + y_power))
        return total


def Fourier_PDE(k1, k2, j1, j2, L, PDE):
  return torch.where(torch.logical_and(k1 == j1, k2 == j2), PDE.evaluate(k1,k2, L), 0.)

def PDE_matrix(m, L, PDE, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
  k1 = k1.flatten()
  k2 = k2.flatten()
  j1 = j1.flatten()
  j2 = j2.flatten()

  PDE_values = Fourier_PDE(k1, k2, j1, j2, L, PDE)

  return PDE_values.view((2*m+1)**2, (2*m+1)**2)


def Omega_matrix(domain, m, device):
  if domain == "square":
    fourier_range = torch.arange(-m, m+1, device=device)
    k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
    k1 = (k1-j1).flatten()
    k2 = (k2-j2).flatten()
    j1, j2 = None, None

    T_values =  torch.mul(torch.sinc(k1/2), torch.sinc(k2/2))/4

    return T_values.view((2*m+1)**2, (2*m+1)**2)
  elif domain == "disk":
    fourier_range = torch.arange(-m, m+1, device=device)
    k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
    k1 = (k1-j1).flatten()
    k2 = (k2-j2).flatten()
    j1, j2 = None, None

    T_values = torch.where(torch.logical_or(k1!= 0, k2 != 0),
                           torch.special.bessel_j1(torch.pi/2*torch.sqrt(k1**2+k2**2))/4/torch.sqrt(k1**2+k2**2), torch.pi/16)
    return T_values.view((2*m+1)**2, (2*m+1)**2)
  

def dimension_m(log10_target_dimension_max):
  m_list=[]
  for i in range(1, int(2*log10_target_dimension_max)+1):
    m_list.append(round((10**(i/4)-1)/2))
  return m_list

def Eigenvalues_numerical(log_m_max, lambda_n, mu_n, s, L, domain, PDE, device):
  is_running_on_gpu()

  m_list = dimension_m(log_m_max)

  spectra, xs = [], []

  for m in m_list:
    P = PDE_matrix(m, L, PDE, device)*(1.0+0*1j)
    T = Omega_matrix(domain, m, device)*(1.0+0*1j)
    PTP = torch.transpose(torch.conj_physical(P), 0, 1)@T@P
    del P

    S = Sob_matrix(m, s, L, device)*(1.0+0*1j)
    M = lambda_n * S + mu_n * PTP
    del S

    Mat = torch.transpose(torch.conj_physical(T), 0, 1)@torch.linalg.solve(M,T)
    del M, T

    eigenvalues = torch.linalg.eigvalsh(Mat)

    x = torch.tensor([i+1 for i in range((2*m+1)**2)])
    sorted_eig = torch.log(torch.abs(eigenvalues))

    xs.append(x)
    spectra.append(sorted_eig)

    print(str((2*m+1)**2)+ " Fourier modes, done.")
  return m_list, xs, spectra

import matplotlib.pyplot as plt

def plot_eigenvalues_2d(m_list, xs, spectra, ymin, ymax):
  mycmap = colors.LinearSegmentedColormap.from_list("", ["lightsteelblue", "royalblue"])
  mycolors = mycmap(np.linspace(0, 1, 10))
  plt.rcParams.update({'font.size': 15})
  for i in range(len(m_list)):
    log10 = torch.log(torch.tensor(10))
    x_log10 = torch.log(xs[i])/log10
    spectra_log10 = torch.sort(spectra[i], descending=True)[0]/log10
    m_log10 = round((torch.log((2*torch.tensor(m_list[i])+1)**2)/log10).item(),1)
    plt.scatter(x_log10.tolist(), spectra_log10.tolist(),
                marker='o', s=10, label="$(2m+1)^2=10^{" + str(m_log10) + "}$",
                color=mycolors[i])
    plt.plot(x_log10.tolist(), spectra_log10.tolist(),
            alpha=0.8,
            color=mycolors[i])


  plt.legend(loc="lower left",fontsize="11")
  plt.xlabel("$\log_{10}$(index)")
  plt.ylabel("$\log_{10}$(Eigenvalue)")
  #plt.title("Eigenvalues decay")
  plt.savefig("2d_fig1.pdf", format="pdf", bbox_inches="tight")
  plt.show()


  for i in range(len(m_list)):
    log10 = torch.log(torch.tensor(10))
    x_log10 = torch.log(xs[i])/log10
    spectra_log10 = torch.sort(spectra[i], descending=True)[0]/log10
    m_log10 = round((torch.log((2*torch.tensor(m_list[i])+1)**2)/log10).item(),1)
    #plt.scatter(x_log10.tolist(), spectra_log10.tolist(), marker='x', s=10, label="log10(m) = "+str(m_log10))
    plt.scatter(x_log10.tolist(), spectra_log10.tolist(),
                marker='o', s=10, label="$(2m+1)^2=10^{" + str(m_log10) + "}$",
                color=mycolors[i])
    plt.plot(x_log10.tolist(), spectra_log10.tolist(),
            alpha=0.8,
            color=mycolors[i])
  plt.legend(loc="upper right",fontsize="11")
  plt.xlabel("$\log_{10}$(index)")
  plt.ylabel("$\log_{10}$(Eigenvalue)")
  #plt.title("Eigenvalues decay")
  plt.ylim(ymin, ymax)
  plt.savefig("2d_fig2.pdf", format="pdf", bbox_inches="tight")
  plt.show()


def Effective_Dimension(log_m_max, log_n_max, lambda_n_exponent, s, L, domain, PDE, device):
  n_list_m = []
  eff_list_m=[]
  m_list = dimension_m(log_m_max)

  for m in m_list:
    print(str((2*m+1)**2)+ " Fourier modes")
    n_list = []
    eff_list = []

    S = Sob_matrix(m, s, L, device)*(1.0+0*1j)
    P = PDE_matrix(m, L, PDE, device)*(1.0+0*1j)
    T = Omega_matrix(domain, m, device)*(1.0+0*1j)

    for N in range(1, 2*log_n_max +1):
      n = int(10**(N/2))
      n_list.append(torch.log(torch.tensor(n)))

      lambda_n = n**(-lambda_n_exponent)
      mu_n = 1/torch.log(torch.tensor(n))
      M = lambda_n * S + mu_n * torch.transpose(torch.conj_physical(P), 0, 1)@T@P

      TMinvT = torch.transpose(torch.conj_physical(T), 0, 1)@torch.linalg.solve(M,T)
      M = None

      Mat = torch.linalg.solve(TMinvT+torch.eye((2*m+1)**2, device=device), TMinvT)
      TMinvT = None

      eff_list.append(torch.trace(Mat))
      Mat = None

      print(str(N/2/log_n_max*100)+"% done")

    n_list, eff_list = torch.tensor(n_list),  torch.real(torch.tensor(eff_list))

    n_list_m.append(n_list), eff_list_m.append(eff_list)
  return m_list, n_list_m, eff_list_m

def plot_effective_dim(m_list, n_list_m, eff_list_m):
  for i in range(len(m_list)):
    log10 = torch.log(torch.tensor(10))
    n_log10 = n_list_m[i]/log10
    eff_log10 = torch.log(eff_list_m[i])/log10
    m_log10 = round((torch.log((2*torch.tensor(m_list[i])+1)**2)/log10).item(),1)

    plt.scatter(n_log10, eff_log10, label = "$\log_{10}$(m) = "+str(m_log10))
    plt.plot(n_log10, eff_log10)
  plt.xlabel("$\log_{10}$(n)")
  plt.ylabel("$\log_{10}$(Effective dimension)")
  plt.legend(loc="lower right")
  plt.savefig("fig3.pdf", format="pdf", bbox_inches="tight")

  plt.show()

def phi_matrix(mat_x, mat_y, mat_j1, mat_j2, L):
  return torch.exp(torch.pi/L*(torch.mul(mat_x, mat_j1)+torch.mul(mat_y, mat_j2))*1j/2)

def M_mat(s, m, lambda_n, mu_n, L, domain, PDE, device):
  S = Sob_matrix(m, s, L, device)*(1.0+0*1j)
  P = PDE_matrix(m, L, PDE, device)*(1.0+0*1j)
  T = Omega_matrix(domain, m, device)*(1.0+0*1j)
  M = lambda_n * S + mu_n * torch.transpose(torch.conj_physical(P), 0, 1)@T@P
  return M

def RFF_fit(data_t, data_x, data_y, s, m, lambda_n, mu_n, L, domain, PDE, device):
  M = M_mat(s, m, lambda_n, mu_n, L, domain, PDE, device)
  return RFF(m, data_t, data_x, data_y, L,  M, device)


def RFF(m, data_t, data_x, data_y, L, M, device):
  l = len(data_x)

  mat_t = torch.tile(data_t, ((2*m+1)**2,1))
  mat_x = torch.tile(data_x, ((2*m+1)**2,1))

  fourier_range = torch.arange(-m, m+1, device=device)
  fourier_rangex = torch.arange(l, device=device)

  j1, j2,  k1 = torch.meshgrid( fourier_range, fourier_range, fourier_rangex, indexing='ij')
  j1 = j1.flatten().view((2*m+1)**2, l)
  j2 = j2.flatten().view((2*m+1)**2, l)

  phi_mat = phi_matrix(mat_t, mat_x, j1, j2, L)

  RFF_mat = phi_mat@torch.conj_physical(torch.transpose(phi_mat, 0, 1))
  data_y = data_y*(1.+0*1j)
  return torch.linalg.solve(RFF_mat+l*M, phi_mat@data_y)



def phi_z_mat(m, data_zt, data_zx, L, device):
  l2 = len(data_zx)
  mat_zt = torch.tile(data_zt, ((2*m+1)**2,1))
  mat_zx = torch.tile(data_zx, ((2*m+1)**2,1))
  fourier_range = torch.arange(-m, m+1, device=device)
  fourier_rangez = torch.arange(l2, device=device)
  jz1, jz2, k1, = torch.meshgrid(fourier_range, fourier_range, fourier_rangez,  indexing='ij')
  jz1 = jz1.flatten().view((2*m+1)**2, l2)
  jz2 = jz2.flatten().view((2*m+1)**2, l2)

  phi_z = phi_matrix(mat_zt, mat_zx, jz1, jz2, L)
  return phi_z


def RFF_estimate(regression_vect, data_zt, data_zx, s, m, n, lambda_n, mu_n, L, domain, PDE, device):
  phi_z = phi_z_mat(m, data_zt, data_zx, L, device)
  estimator = torch.transpose(torch.conj_physical(phi_z), 0,1)@regression_vect
  return estimator



dX = DifferentialOperator({(1, 0): 1})
dY = DifferentialOperator({(0, 1): 1})