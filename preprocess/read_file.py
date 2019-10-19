from tqdm import tqdm

import global_vars
source_path = global_vars.source_path
file_size = global_vars.file_size



def read_file(file_name):
  file = open(source_path + file_name + "_sample", "r")
  signal = []

  for idx in tqdm(range(file_size), desc="READING", ncols=100, unit=" signal"):
    line = file.readline().rstrip(" \n")
    data = line.split(" ")
    sample = [float(i) for i in data]
    signal.append(sample)

  file.close()
  return signal
