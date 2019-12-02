import numpy as np
from global_vars import *
from tqdm import tqdm



def test_sample(test, answer):
  try:
    count = 0
    if sample_type == "org":
      start_idx = 12 * repetition
    elif sample_type == "128bit":
      start_idx = 0

    for idx in range(start_idx, len(test), repetition):
      unit_count = 0
      for unit_idx in range(repetition):
        if (test[idx+unit_idx] < 0.5 and answer[idx+unit_idx] == 0) or (test[idx+unit_idx] >= 0.5 and answer[idx+unit_idx] == 1):
          unit_count += 1
      if unit_count / repetition > 0.5:
        count += 1
    return count

  except Exception as ex:
    print("[test_set.py: test_sample]", end=" ")
    print(ex)



def test_set(file_name, test, answer):
  try:
    success = 0
    success_bit = np.zeros(257)

    for idx in tqdm(range(len(test)), desc="TESTING", ncols=80, unit="signal"):
      count = test_sample(test[idx], answer[idx])
      success_bit[count] += 1
      if count == 256:
        success += 1

    file = open(log_full_path, "a")
    file.write("\t\t\t***** " + file_name + " *****\n\n")
    file.write(str(success) + " / " + str(len(test)))
    file.write(" (" + str(round(100 * (float(success) / len(test)), 2)) + "%)\n\n")
    for idx in range(257):
      file.write(str(int(success_bit[idx])) + " ")

    ber = 0
    for idx in range(257):
      ber += idx * success_bit[idx]
    ber = 1 - ber / (len(test) * 256)
    file.write("\n\nBER: " + str(ber) + "\n\n")
    file.close()

    return success, success_bit, ber

  except Exception as ex:
    print("[test_set.py]", end=" ")
    print(ex)
