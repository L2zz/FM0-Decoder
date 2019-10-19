from tqdm import tqdm

import global_vars
data_path = global_vars.data_path
file_size = global_vars.file_size



def read_file(file_name):
  try:
    file_sample = open(data_path + file_name + "_sample", "r")
    file_databit = open(data_path + file_name + "_databit", "r")
    signal = []
    databit = []

    for idx in tqdm(range(file_size), desc="READING", ncols=100, unit=" signal"):
      line = file_sample.readline().rstrip(" \n")
      data = line.split(" ")
      sample = [float(i) for i in data]
      signal.append(sample)

      line = file_databit.readline().rstrip("\n")
      data = [int(i) for i in line]
      databit.append(data)

    file_sample.close()
    file_databit.close()
    return signal, databit

  except Exception as ex:
    print("[read_file.py]", end=" ")
    print(ex)
