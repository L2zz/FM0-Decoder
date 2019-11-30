import numpy as np
from tqdm import tqdm

import global_vars
tail = global_vars.tail
log_full_path = global_vars.log_full_path



def test_sample(test, answer, unit):
  try:
    count = 0
    #for idx in range(0, len(test), unit):
    for idx in range(12*unit, len(test), unit):
      unit_count = 0
      for unit_idx in range(unit):
        if (test[idx+unit_idx] < 0.5 and answer[idx+unit_idx] == 0) or (test[idx+unit_idx] >= 0.5 and answer[idx+unit_idx] == 1):
          unit_count += 1
      if unit_count / unit > 0.5:
        count += 1
    return count

  except Exception as ex:
    print("[test_sample.py]", end=" ")
    print(ex)



def test_set(test, answer):
  if tail == "_enc256":
    try:
      success = 0
      success_bit = np.zeros(257)

      for idx in tqdm(range(len(test)), desc="TESTING", ncols=80, unit="signal"):
        count = test_sample(test[idx], answer[idx], 1)
        success_bit[count] += 1
        if count == 256:
          success += 1

      file = open(log_full_path, "a")
      file.write(str(success) + " / " + str(len(test)) + "\n\n")
      for idx in range(257):
        file.write(str(int(success_bit[idx])) + " ")

      ber = 0
      for idx in range(257):
        ber += idx * success_bit[idx]
      ber = 1 - ber / (len(test) * 256)
      file.write("\n\nBER: " + str(ber))

      return success, ber

    except Exception as ex:
      print("[test_set.py]", end=" ")
      print(ex)



  if tail == "_enc256_5":
    try:
      success = 0
      success_bit = np.zeros(257)

      for idx in tqdm(range(len(test)), desc="TESTING", ncols=80, unit="signal"):
        count = test_sample(test[idx], answer[idx], 5)
        success_bit[count] += 1
        if count == 256:
          success += 1

      file = open(log_full_path, "a")
      file.write(str(success) + " / " + str(len(test)) + "\n\n")
      for idx in range(257):
        file.write(str(int(success_bit[idx])) + " ")

      return success

    except Exception as ex:
      print("[test_set.py]", end=" ")
      print(ex)
