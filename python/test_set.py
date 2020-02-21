import sys
import numpy as np
from global_vars import *
from tqdm import tqdm

sys.path.append("/home/l2zz/mnt/sin-decode/comparative/correlation_based/python")
from decode_data import decode_data

def test_sample(test, answer):
  try:
    count = 0

    add_iq = test.sum(axis=1)
    decoded_bit = decode_data(add_iq)

    for bit_idx in range(bit_data):
      if decoded_bit[bit_idx] == answer[bit_idx]:
          count += 1

    return count

  except Exception as ex:
    print("[test_set.py: test_sample]", end=" ")
    print(ex)



def test_set(file_name, test, answer):
  try:
    success = 0
    success_bit = np.zeros(129)

    for idx in tqdm(range(len(test)), desc="TESTING", ncols=80, unit="signal"):
      count = test_sample(test[idx][0], answer[idx])
      success_bit[count] += 1
      if count == 128:
        success += 1

    file = open(log_full_path, "a")
    file.write("\t\t\t***** " + file_name + " *****\n\n")
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
