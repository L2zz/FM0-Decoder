import random

from global_vars import *
from SignalSet import SignalSet



def make_set(input_signals, index_set):
  try:
    input_set = SignalSet()

    for idx in index_set.train:
      input_set.train.append(input_signals[idx])

    for idx in index_set.validation:
      input_set.validation.append(input_signals[idx])

    for idx in index_set.test:
      input_set.test.append(input_signals[idx])

    return input_set

  except Exception as ex:
    print("[make_set.py]", end=" ")
    print(ex)
