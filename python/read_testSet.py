from global_vars import *
from tqdm import tqdm



def read_testSet(file_name):
  try:
    file = open(data_path + file_name + "_sample" + RN_tail, "r")
    input_signals = []

    for n in tqdm(range(test_size), desc=file_name, ncols=80, unit="signal"):
      line = file.readline().rstrip(" \n").split(" ")
      data = [[float(i)] for i in line]
      input_signals.append(data)

    file.close()

    if repetition == 1:
      file = open(data_path + file_name + "_answer" + tail + RN_tail, "r")
    else:
      file = open(data_path + file_name + "_answer" + tail + "_" + repetition + RN_tail, "r")
    answer_signals = []

    for n in range(test_size):
      line = file.readline().rstrip("\n")
      data = [int(i) for i in line]
      answer_signals.append(data)

    file.close()
    return input_signals, answer_signals

  except Exception as ex:
    print("[read_testSet.py: " + file_name + "]", end=" ")
    print(ex)
