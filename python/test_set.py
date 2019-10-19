import numpy as np
from tqdm import tqdm

import global_vars
tail = global_vars.tail
log_path = global_vars.log_path



def test_sample(test, answer):
  try:
    count = 0
    for idx in range(len(test)):
      if (test[idx] < 0.5 and answer[idx] == 0) or (test[idx] >= 0.5 and answer[idx] == 1):
        count += 1
    return count

  except Exception as ex:
    print("[test_sample.py]", end=" ")
    print(ex)



def test_set(test, answer):
  try:
    success = 0
    success_bit = np.zeros(len(test[0]) + 1)

    for idx in tqdm(range(len(test)), desc="TESTING", ncols=80, unit="signal"):
      count = test_sample(test[idx], answer[idx])
      success_bit[count] += 1
      if count == len(test[0]):
        success += 1

    file = open(log_path, "w")
    file.write(str(success) + " / " + str(len(test)) + "\n\n")
    for idx in range(len(test[0]) + 1):
      file.write(str(int(success_bit[idx])) + " ")

    return success

  except Exception as ex:
    print("[test_set.py]", end=" ")
    print(ex)
