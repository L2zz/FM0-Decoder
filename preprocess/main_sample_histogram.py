import numpy as np
import matplotlib.pyplot as plt
from math import *

import global_vars
target_path = global_vars.target_path
file_size = global_vars.file_size
file_name_list = global_vars.file_name_list



# https://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins



if __name__ == "__main__":
  file = open(target_path + "average_std", "r")
  tot = []

  line = file.readline()
  for file_name in file_name_list:
    line = file.readline()
    line = file.readline()
    line = file.readline().rstrip(" \n").split(" ")
    data = [float(i) for i in line]
    tot.extend(data)
    '''
    plt.title(file_name)
    plt.hist(data, rwidth=0.8)
    plt.savefig(target_path + "p_" + file_name + ".png")
    plt.clf()
    '''
  file.close()
  file = open(target_path + "variation_std", "r")
  tot2 = []

  line = file.readline()
  for file_name in file_name_list:
    line = file.readline()
    line = file.readline()
    line = file.readline().rstrip(" \n").split(" ")
    data = [sqrt(float(i)) for i in line]
    tot2.extend(data)

  plt.scatter(tot, tot2)
  plt.xlabel("average")
  plt.ylabel("std_dev")
  #plt.hist(tot2, rwidth=0.8)
  #plt.plot(data)
  #plt.hist(tot, compute_histogram_bins(tot2, 1e-3), rwidth=0.8)
  plt.show()
  file.close()
