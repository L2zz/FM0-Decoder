import timeit

class ExecutionTime:
  def __init__(self, name):
    self.name = name
    self.time = timeit.default_timer()

  def print(self):
    print("\t" + self.name + " EXECUTION TIME:\t" + str(round(self.time, 3)) + " (sec)")

  def print_noname(self):
    print("\tEXECUTION TIME:\t" + str(round(self.time, 3)) + " (sec)")

  def print_file(self, file):
    file.write("EXECUTION TIME:\t" + str(round(self.time, 3)) + " (sec)\n\n")

  def stop(self, is_name):
    self.time = timeit.default_timer() - self.time
    if is_name: self.print()
    else: self.print_noname()
