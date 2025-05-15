
import torch
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from pikernel.utils import *

def Sob_formula_1d(k, j, s, L):
    return torch.where(k == j, 1+ k**(2*s)/(2*L)**s, 0.)


def Sob_matrix_1d(m, s, L, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k, j = torch.meshgrid(fourier_range, fourier_range, indexing='ij')
  k = k.flatten()
  j = j.flatten()

  sob_values = Sob_formula_1d(k, j, s, L)

  return sob_values.view(2*m+1, 2*m+1)


class DifferentialOperator1d:
    def __init__(self, coefficients=None):
        """
        Initialize the PDE.
        The keys are tuples representing the powers of d/dX.
        For example, {(2): 3, (0): -1} represents 3d^2/dX^2 - 1.
        """
        if coefficients is None:
            self.coefficients = {}
        else:
            self.coefficients = coefficients

    def __repr__(self):
        terms = []
        for (x_power), coefficient in sorted(self.coefficients.items(), reverse=True):
            if coefficient == 0:
                continue
            term = f"{coefficient}"
            if x_power != 0:
                term += f"*(d/dX)^{x_power}"
            terms.append(term)
        PDE = " + ".join(terms) if terms else "0"
        return "The PDE of your model is " + PDE + " = 0."

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = DifferentialOperator1d(self.coefficients.copy())
            if (0) in result.coefficients:
                result.coefficients[(0)] += other
            else:
                result.coefficients[(0)] = other
            return result

        result = DifferentialOperator1d(self.coefficients.copy())
        for (x_power), coefficient in other.coefficients.items():
            if (x_power) in result.coefficients:
                result.coefficients[(x_power)] += coefficient
            else:
                result.coefficients[(x_power)] = coefficient
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        result = DifferentialOperator1d(self.coefficients.copy())
        for (x_power), coefficient in other.coefficients.items():
            if (x_power) in result.coefficients:
                result.coefficients[(x_power)] -= coefficient
            else:
                result.coefficients[(x_power)] = -coefficient
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = DifferentialOperator1d()
            for (x_power), coefficient in self.coefficients.items():
                result.coefficients[(x_power)] = coefficient * other
            return result

        result = DifferentialOperator1d()
        for (x1), c1 in self.coefficients.items():
            for (x2), c2 in other.coefficients.items():
                power = (x1 + x2)
                if power in result.coefficients:
                    result.coefficients[power] += c1 * c2
                else:
                    result.coefficients[power] = c1 * c2
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        if exponent == 0:
            return DifferentialOperator1d({(0): 1})
        elif exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")

        result = DifferentialOperator1d(self.coefficients.copy())
        for _ in range(1, exponent):
            result *= self
        return result

    def evaluate(self, x, L):
        total = 0
        geometry = -1j*torch.pi/2/L
        for (x_power), coefficient in self.coefficients.items():
            total += coefficient * (x ** x_power) * (geometry **(x_power))
        return total


def Fourier_PDE_1d(k, j, L, PDE):
  return torch.where(k == j, PDE.evaluate(k, L), 0.)

def PDE_matrix_1d(m, L, PDE, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k, j = torch.meshgrid(fourier_range, fourier_range, indexing='ij')
  k = k.flatten()
  j = j.flatten()

  PDE_values = Fourier_PDE_1d(k, j, L, PDE)

  return PDE_values.view(2*m+1, 2*m+1)

def Omega_matrix_1d(m, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k, j = torch.meshgrid(fourier_range, fourier_range, indexing='ij')
  k = (k-j).flatten()
  j = None

  T_values = torch.where(k == 0, 1/2, 0.) + torch.where(k != 0, torch.sin(torch.pi*k/2)/k/torch.pi, 0.)


  return T_values.view(2*m+1, 2*m+1)




def dimension_m_1d(log10_target_dimension_max):
  m_list=[]
  for i in range(1, int(2*log10_target_dimension_max)+1):
    m_list.append(round((10**(i/2)-1)/2))
  return m_list

def Eigenvalues_numerical_1d(log_m_max, lambda_n, mu_n, s, L, PDE, device):
  is_running_on_gpu()

  m_list = dimension_m_1d(log_m_max)

  spectra, xs = [], []

  for m in m_list:
    P = PDE_matrix_1d(m, L, PDE, device)*(1.0+0*1j)
    T = Omega_matrix_1d(m, device)*(1.0+0*1j)
    PTP = torch.transpose(torch.conj_physical(P), 0, 1)@T@P
    del P

    S = Sob_matrix_1d(m, s, L, device)*(1.0+0*1j)
    M = lambda_n * S + mu_n * PTP
    del S

    Mat = torch.transpose(torch.conj_physical(T), 0, 1)@torch.linalg.solve(M,T)
    del M, T

    eigenvalues = torch.linalg.eigvalsh(Mat)

    x = torch.tensor([i+1 for i in range(2*m+1)])
    sorted_eig = torch.log(torch.abs(eigenvalues))

    xs.append(x)
    spectra.append(sorted_eig)

    print(str(2*m+1)+ " Fourier modes, done.")
  return m_list, xs, spectra



def plot_eigenvalues_1d(m_list, xs, spectra, ymin, ymax, theoretical_bound):
  mycmap = colors.LinearSegmentedColormap.from_list("", ["lightsteelblue", "royalblue"])
  #blues = cm.get_cmap(mycmap, 10)
  blues = mycmap(np.linspace(0, 1, 10))
  plt.rcParams.update({'font.size': 15})
  for i in range(len(m_list)):
    log10 = torch.log(torch.tensor(10))
    x_log10 = torch.log(xs[i])/log10
    spectra_log10 = torch.sort(spectra[i], descending=True)[0]/log10
    m_log10 = round((torch.log(2*torch.tensor(m_list[i])+1)/log10).item(),1)
    plt.scatter(x_log10.tolist(), spectra_log10.tolist(),
                marker='o', s=10, label="$2m+1=10^{" + str(m_log10) + "}$", alpha=0.8,
                color=blues[i]) #label="log10(m) = "+str(m_log10).  #r'm=$10^{}$'.format(m_log10)
    plt.plot(x_log10.tolist(), spectra_log10.tolist(),
            alpha=0.8,
            color=blues[i])
  if theoretical_bound:
    interval = xs[-1].numpy()[3:]
    upper_bound = 4/(lambda_n + mu_n)/(interval-2)**2
    plt.scatter(np.log(interval)/np.log(10), np.log(upper_bound)/np.log(10),
                marker='+', s=20, label="Theoretical UB", color='lightsalmon', alpha=0.8)
  plt.legend(loc='lower left', fontsize="11") #bbox_to_anchor=(1.05, 0.5)
  plt.xlabel("$\log_{10}$(index)")
  plt.ylabel("$\log_{10}$(Eigenvalue)")
  #plt.title("Eigenvalues decay")

  plt.savefig("fig1.pdf", format="pdf", bbox_inches="tight")
  plt.show()


  plt.rcParams.update({'font.size': 15})
  for i in range(len(m_list)):
    log10 = torch.log(torch.tensor(10))
    x_log10 = torch.log(xs[i])/log10
    spectra_log10 = torch.sort(spectra[i], descending=True)[0]/log10
    m_log10 = round((torch.log(2*torch.tensor(m_list[i])+1)/log10).item(),1)
    plt.scatter(x_log10.tolist(), spectra_log10.tolist(),
                marker='o', s=10, label="$2m+1=10^{" + str(m_log10) + "}$", alpha=0.8,
                color=blues[i])
    plt.plot(x_log10.tolist(), spectra_log10.tolist(),
            alpha=0.8,
            color=blues[i])
  if theoretical_bound:
    interval = xs[-1].numpy()[3:]
    upper_bound = 4/(lambda_n + mu_n)/(interval-2)**2
    plt.scatter(np.log(interval)/np.log(10), np.log(upper_bound)/np.log(10),
                marker='+', s=20, label="Theoretical UB", alpha=0.8, color='lightsalmon')
  plt.legend(loc="upper right",fontsize="11")
  plt.xlabel("$\log_{10}$(index)")
  plt.ylabel("$\log_{10}$(Eigenvalue)")
  #plt.title("Eigenvalues decay")
  plt.ylim(ymin, ymax)

  plt.savefig("fig2.pdf", format="pdf", bbox_inches="tight")
  plt.show()





def Effective_Dimension_1d(log_m_max, log_n_max, lambda_n_exponent, s, L, PDE, device):
  n_list_m = []
  eff_list_m=[]
  m_list = dimension_m_1d(log_m_max)

  for m in m_list:
    print(str(2*m+1)+ " Fourier modes")
    n_list = []
    eff_list = []

    S = Sob_matrix_1d(m, s, L, device)*(1.0+0*1j)
    P = PDE_matrix_1d(m, L, PDE, device)*(1.0+0*1j)
    T = Omega_matrix_1d(m, device)*(1.0+0*1j)

    for N in range(1, 2*log_n_max +1):
      n = int(10**(N/2))
      n_list.append(torch.log(torch.tensor(n)))

      lambda_n = n**(-lambda_n_exponent)
      mu_n = 1/torch.log(torch.tensor(n))
      M = lambda_n * S + mu_n * torch.transpose(torch.conj_physical(P), 0, 1)@T@P

      TMinvT = torch.transpose(torch.conj_physical(T), 0, 1)@torch.linalg.solve(M,T)
      M = None

      Mat = torch.linalg.solve(TMinvT+torch.eye(2*m+1, device=device), TMinvT)
      TMinvT = None

      eff_list.append(torch.trace(Mat))
      Mat = None

      print(str(N/2/log_n_max*100)+"% done")

    n_list, eff_list = torch.tensor(n_list),  torch.real(torch.tensor(eff_list))

    n_list_m.append(n_list), eff_list_m.append(eff_list)
  return m_list, n_list_m, eff_list_m

def plot_effective_dim_1d(m_list, n_list_m, eff_list_m):
  log10 = torch.log(torch.tensor(10))
  n_log10 = n_list_m[-1]/log10
  plt.rcParams.update({'font.size': 15})
  eff_log10 = torch.log(eff_list_m[-1])/log10
  m_log10 = round((torch.log(2*torch.tensor(m_list[-1])+1)/log10).item(),1)

  plt.rcParams.update({'font.size': 15})
  plt.scatter(n_log10, eff_log10,color=myLightGreen, s=40,
              label = "$2m+1=10^{" + str(m_log10) + "}$", marker='d')
  plt.plot(n_log10, eff_log10,color=myLightGreen, linewidth=2)
  plt.xlabel("$\log_{10}$(n)")
  plt.ylabel("$\log_{10}$(Effective dimension)")
  #plt.title("Effective dimension N with log(m) = "+str(m_log10))
  plt.legend(loc="lower right")

  plt.savefig("effective_dim_"+str(m_log10)+".pdf", format="pdf", bbox_inches="tight")
  plt.show()

def plot_effective_dim_1d2(m_list, n_list_m, eff_list_m):
  mycmap = colors.LinearSegmentedColormap.from_list("", ["khaki", myLightGreen, myDarkGreen])
  mycolors = mycmap(np.linspace(0, 1, 10))
  plt.rcParams.update({'font.size': 15})
  log10 = torch.log(torch.tensor(10))
  n_log10 = (torch.round(n_list_m[0]/log10, decimals=1)).tolist()
  m_log10 = torch.round((torch.log(2*torch.tensor(m_list)+1)/log10),decimals=1)#.item()
  m = len(m_list)

  for i in range(len(n_list_m)):
    eff_dim_n = [torch.log(eff_list_m[j])[i]/log10 for j in range(m)]
    plt.scatter(m_log10, eff_dim_n, label = "$n=10^{" + str(n_log10[i]) + "}$",
                color=mycolors[i], marker='d', s=40)
    plt.plot(m_log10, eff_dim_n, color=mycolors[i], linewidth=2)

  plt.xlabel("$\log_{10}$(2m+1)")
  plt.ylabel("$\log_{10}$(Effective dimension)")
  #plt.title("Effective dimension N")
  plt.legend(loc="lower right",ncol=2,fontsize="11")

  plt.savefig("effective_dimension_n.pdf", format="pdf", bbox_inches="tight")
  plt.show()



def plot_effective_dim_2d(m_list, n_list_m, eff_list_m):
  log10 = torch.log(torch.tensor(10))
  n_log10 = n_list_m[-1]/log10
  plt.rcParams.update({'font.size': 15})
  eff_log10 = torch.log(eff_list_m[-1])/log10
  m_log10 = round((torch.log(2*torch.tensor(m_list[-1])+1)/log10).item(),1)

  plt.rcParams.update({'font.size': 15})
  plt.scatter(n_log10, eff_log10,color=myLightGreen, s=40,
              label = "$(2m+1)^2=10^{" + str(2*m_log10) + "}$", marker='d')
  plt.plot(n_log10, eff_log10,color=myLightGreen, linewidth=2)
  plt.xlabel("$\log_{10}$(n)")
  plt.ylabel("$\log_{10}$(Effective dimension)")
  #plt.title("Effective dimension N with log(m) = "+str(m_log10))
  plt.legend(loc="lower right")

  plt.savefig("effective_dim_"+str(m_log10)+".pdf", format="pdf", bbox_inches="tight")
  plt.show()

def plot_effective_dim_2d2(m_list, n_list_m, eff_list_m):
  mycmap = colors.LinearSegmentedColormap.from_list("", ["khaki", myLightGreen, myDarkGreen])
  mycolors = mycmap(np.linspace(0, 1, 10))
  plt.rcParams.update({'font.size': 15})
  log10 = torch.log(torch.tensor(10))
  n_log10 = (torch.round(n_list_m[0]/log10, decimals=1)).tolist()
  m_log10 = torch.round((torch.log(2*torch.tensor(m_list)+1)/log10),decimals=1)#.item()
  m = len(m_list)

  for i in range(len(n_list_m)):
    eff_dim_n = [torch.log(eff_list_m[j])[i]/log10 for j in range(m)]
    plt.scatter(2*m_log10, eff_dim_n, label = "$n=10^{" + str(n_log10[i]) + "}$",
                color=mycolors[i], marker='d', s=40)
    plt.plot(2*m_log10, eff_dim_n, color=mycolors[i], linewidth=2)

  plt.xlabel("$2\log_{10}$(2m+1)")
  plt.ylabel("$\log_{10}$(Effective dimension)")
  #plt.title("Effective dimension N")
  plt.legend(loc="lower right",ncol=2,fontsize="11")

  plt.savefig("effective_dimension_n.pdf", format="pdf", bbox_inches="tight")
  plt.show()

def phi_matrix_1d(mat_x, mat_j, L):
  return torch.exp(torch.pi/L*(torch.mul(mat_x, mat_j))*1j/2)

def M_mat_1d(s, m, lambda_n, mu_n, L, PDE, device):
  S = Sob_matrix_1d(m, s, L, device)*(1.0+0*1j)
  P = PDE_matrix_1d(m, L, PDE, device)*(1.0+0*1j)
  T = Omega_matrix_1d(m, device)*(1.0+0*1j)
  M = lambda_n * S + mu_n * torch.transpose(torch.conj_physical(P), 0, 1)@T@P
  return M

def RFF_fit_1d(data_x, data_y, s, m, lambda_n, mu_n, L, PDE, device):
  M = M_mat_1d(s, m, lambda_n, mu_n, L, PDE, device)
  return RFF_1d(m, data_x, data_y, L,  M, device)


def RFF_1d(m, data_x, data_y, L, M, device):
  l = len(data_x)

  mat_x = torch.tile(data_x, (2*m+1,1))

  fourier_range = torch.arange(-m, m+1, device=device)
  fourier_rangex = torch.arange(l, device=device)

  j1,  k1 = torch.meshgrid( fourier_range, fourier_rangex, indexing='ij')
  j1 = j1.flatten().view(2*m+1, l)

  phi_mat = phi_matrix_1d(mat_x, j1, L)

  RFF_mat = phi_mat@torch.conj_physical(torch.transpose(phi_mat, 0, 1))
  data_y = data_y*(1.+0*1j)
  return torch.linalg.solve(RFF_mat+l*M, phi_mat@data_y)



def phi_z_mat_1d(m, data_zt, L, device):
  l2 = len(data_zt)
  mat_zt = torch.tile(data_zt, (2*m+1,1))
  fourier_range = torch.arange(-m, m+1, device=device)
  fourier_rangez = torch.arange(l2, device=device)
  jz1, k1, = torch.meshgrid(fourier_range, fourier_rangez,  indexing='ij')
  jz1 = jz1.flatten().view(2*m+1, l2)

  phi_z = phi_matrix_1d(mat_zt, jz1, L)
  return phi_z


def RFF_estimate_1d(regression_vect, data_zt, s, m, n, lambda_n, mu_n, L, PDE, device):
  phi_z = phi_z_mat_1d(m, data_zt, L, device)
  estimator = torch.transpose(torch.conj_physical(phi_z), 0,1)@regression_vect
  return estimator



dX = DifferentialOperator1d({(1): 1})
