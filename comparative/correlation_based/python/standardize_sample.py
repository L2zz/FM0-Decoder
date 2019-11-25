import math
from tqdm import tqdm

import global_vars
file_size = global_vars.file_size



def standardize_sample(signal):
  try:
    new_signal = []
    sample_size = len(signal[0])

    for idx in tqdm(range(file_size), desc="STANDARDIZE", ncols=100, unit=" signal"):
      avg = 0
      for sample in signal[idx]:
        avg += sample
      avg /= sample_size

      std = 0
      for sample in signal[idx]:
        std += pow(sample - avg, 2)
      std /= sample_size
      std = math.sqrt(std)

      new_sample = []
      for sample in signal[idx]:
        value = (sample - avg) / std
        #if value > 1: new_sample.append(float(1))
        #elif value < -1: new_sample.append(float(-1))
        #else: new_sample.append(value)
        new_sample.append(value)

      new_signal.append(new_sample)

    return new_signal

  except Exception as ex:
    print("[standardize_sample.py]", end=" ")
    print(ex)
