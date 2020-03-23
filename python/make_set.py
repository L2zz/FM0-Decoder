import random

from global_vars import *
from SignalSet import SignalSet



def make_set(input_signals, answer_signals, answer_bits, index_set):
  try:
    input_set = SignalSet()
    answer_set = SignalSet()
    answer_bit_set = SignalSet()

    for idx in index_set.train:
      input_set.train.append(input_signals[idx])
      answer_set.train.append(answer_signals[idx])

    for idx in index_set.validation:
      input_set.validation.append(input_signals[idx])
      answer_set.validation.append(answer_signals[idx])

    for idx in index_set.test:
      input_set.test.append(input_signals[idx])
      answer_set.test.append(answer_signals[idx])
      answer_bit_set.test.append(answer_bits[idx])

    return input_set, answer_set, answer_bit_set

  except Exception as ex:
    print("[make_set.py]", end=" ")
    print(ex)
