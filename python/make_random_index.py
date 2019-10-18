import random

def make_random_index(size):
  try:
    random_index = []

    while len(random_index) != size:
      index = random.randrange(0, size)
      if index not in random_index:
        random_index.append(index)

    return random_index

  except Exception as ex:
    print("[make_random_index.py]", end=" ")
    print(ex)
