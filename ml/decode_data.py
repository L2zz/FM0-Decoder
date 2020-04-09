import numpy as np
from tqdm import tqdm

from global_vars import *

def _decode_enc256(input, enc256):
    count = 0
    for idx in range(12, len(input), 2):
        if input[idx] < 0.5:
            half_bit1 = 0
        else:
            half_bit1 = 1
        if input[idx+1] < 0.5:
            half_bit2 = 0
        else:
            half_bit2 = 1

        if enc256[idx] == half_bit1 and enc256[idx+1] == half_bit2:
            count += 1

    return count

def decode_enc256(input, enc256):
    success = 0
    success_bit = np.zeros(129)

    for idx in tqdm(range(len(input)), desc="TESTING", ncols=80, unit="signal"):
      count = _decode_enc256(input[idx], enc256[idx])
      success_bit[count] += 1
      if count == 128:
        success += 1

    file = open(log_file_path, "a")
    file.write(str(success) + " / " + str(len(input)))
    file.write(" (" + str(round(100 * (float(success) / len(input)), 2)) + "%)\n\n")
    for idx in range(129):
      file.write(str(int(success_bit[idx])) + " ")

    ber = 0
    for idx in range(129):
      ber += idx * success_bit[idx]
    ber = 1 - ber / (len(input) * 128)
    file.write("\n\nBER: " + str(ber) + "\n\n")
    file.close()

    return success, success_bit, ber
