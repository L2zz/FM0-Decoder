import timeit

class ExecutionTime:
  def __init__(self, name):
    self.name = name
    self.time = timeit.default_timer()

  def print(self):
    print("\t" + self.name + " EXECUTION TIME:\t" + str(round(self.time, 3)) + " (sec)")

  def stop(self, is_print):
    self.time = timeit.default_timer() - self.time

    if is_print:
      print("\n\n\n")
      self.print()
      print("\n\n")
