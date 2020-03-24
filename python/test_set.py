import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from global_vars import *
from tqdm import tqdm

sys.path.append("/home/l2zz/mnt/sin-decode/comparative/correlation_based/python")
from decode_data import decode_data

def test_sample(test, answer):
  try:
    count = 0

    # avg_iq = test
    #
    # avg = 0
    # for iq in avg_iq:
    #     avg += iq
    # avg /= len(avg_iq)
    #
    # std = 0
    # for iq in avg_iq:
    #     std += pow(iq - avg, 2)
    # std /= len(avg_iq)
    # std = math.sqrt(std)
    #
    # std_avg_iq = []
    # for iq in avg_iq:
    #     val = (iq - avg) / std
    #     if val > 1: std_avg_iq.append(float(1))
    #     elif val < -1: std_avg_iq.append(float(-1))
    #     else: std_avg_iq.append(val)

    decoded_bit = decode_data(test)

    for bit_idx in range(bit_data):
      if decoded_bit[bit_idx] == answer[bit_idx]:
          count += 1

    return count

  except Exception as ex:
    print("[test_set.py: test_sample]", end=" ")
    print(ex)



def test_set(test, answer):
  try:
    success = 0
    success_bit = np.zeros(129)

    for idx in tqdm(range(len(test)), desc="TESTING", ncols=80, unit="signal"):
      count = test_sample(test[idx][0], answer[idx])
      success_bit[count] += 1
      if count == 128:
        success += 1

    file = open(log_full_path, "a")
    file.write("\t\t\t***** Summary *****\n\n")
    file.write(str(success) + " / " + str(len(test)))
    file.write(" (" + str(round(100 * (float(success) / len(test)), 2)) + "%)\n\n")
    for idx in range(129):
      file.write(str(int(success_bit[idx])) + " ")

    ber = 0
    for idx in range(129):
      ber += idx * success_bit[idx]
    ber = 1 - ber / (len(test) * 128)
    file.write("\n\nBER: " + str(ber) + "\n\n")
    file.close()

    return success, success_bit, ber

  except Exception as ex:
    print("[test_set.py]", end=" ")
    print(ex)
