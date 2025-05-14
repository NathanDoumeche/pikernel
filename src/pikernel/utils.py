import torch
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)

def is_running_on_gpu():
  if torch.cuda.is_available():
    print("The algorithm is running on GPU.")
  else:
    print("The algorithm is not running on GPU.")


def save(list1, list2, list3, name):
  pd.DataFrame(data={"a": list1, "b": [i.tolist() for i in list2], "c": [i.tolist() for i in list3]}).to_csv(name+".csv")
  return

def str_to_list(s):
  list_s = s[1:len(s)-1].split(',')
  return torch.tensor([float(i) for i in list_s])

def read(path):
  data = pd.read_csv(path)
  m_list, n_list_m, eff_list_m = torch.tensor(data['a']), data['b'], data['c']
  n_list_m = [str_to_list(n_list_m[i]) for i in range(len(n_list_m))]
  eff_list_m = [str_to_list(eff_list_m[i]) for i in range(len(eff_list_m))]
  return m_list, n_list_m, eff_list_m



myDarkOrange = colors.to_rgb('#F47A1F')
myLightOrange = colors.to_rgb('#FDBB2F')
myDarkGreen = colors.to_rgb('#00743F')
myLightGreen = colors.to_rgb('#25B396')
myTurquoise = colors.to_rgb('#70CED0')
myLightBlue = colors.to_rgb('#1E65A7')
myDarkBlue = colors.to_rgb('#192E5B')


