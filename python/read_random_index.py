from global_vars import *
from SignalSet import SignalSet



def read_random_index(index_set):
  try:
    file = open(RN_path, "r")

    line = file.readline().rstrip(" \n").split(" ")
    index_set.train = [int(i) for i in line]
    line = file.readline().rstrip(" \n").split(" ")
    index_set.validation = [int(i) for i in line]
    line = file.readline().rstrip(" \n").split(" ")
    index_set.test = [int(i) for i in line]
    index_set.test.sort()
    

    file.close()

  except Exception as ex:
    print("[read_random_index.py]", end=" ")
    print(ex)
