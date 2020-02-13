from global_vars import *
from tqdm import tqdm



def read_file(file_name):
  try:
    file = open(data_path + file_name + "_sample", "r")
    input_signals = []

    for n in tqdm(range(file_size), desc=file_name, ncols=80, unit="signal"):
      line = file.readline().rstrip(" \n")
      if not line: break
      data = line.split(" ")
      data = [[float(i)] for i in data]
      input_signals.append(data)

    file.close()

    if repetition == 1:
      file = open(data_path + file_name + "_answer" + tail, "r")
    else:
      file = open(data_path + file_name + "_answer" + tail + "_" + repetition, "r")
    answer_signals = []

    for n in range(file_size):
      line = file.readline().rstrip("\n")
      if not line: break
      data = [int(i) for i in line]
      answer_signals.append(data)

    file.close()
    return input_signals, answer_signals

  except Exception as ex:
    print("[read_file.py: " + file_name + "]", end=" ")
    print(ex)
