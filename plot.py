import os
from plot_log import plot_log

import global_vars
log_path = global_vars.log_path



def read_file(log_name):
  file = open(log_path + log_name, "r")

  line = file.readline().rstrip("\n")
  if line == "loss":
    line = file.readline().rstrip("]\n")
    line = line.replace("[", "")
    line = line.split(", ")
    loss = [float(i) for i in line]

    file.readline()
    file.readline()
    line = file.readline().rstrip("]\n")
    line = line.replace("[", "")
    line = line.split(", ")
    val_loss = [float(i) for i in line]

    file.readline()
    line = file.readline().rstrip("\n")
    type = True
  else:
    loss = []
    val_loss = []
    type = False

  line = line.split(" / ")
  line = [int(i) for i in line]
  success = float(line[0]) / line[1]

  file.close()
  return loss, val_loss, success, type



if __name__ == "__main__":
  log_name_list = []

  try:
    print("\n\n\n" + str(os.listdir(log_path)) + "\n\n\n")

    while True:
      try:
        log_name = input("Input the log name you want to plot (press \"x\" to continue): ").rstrip("\n")
        if log_name == 'X' or log_name == 'x':
          break
        if not os.path.exists(log_path + log_name):
          raise NameError("Log file \"" + log_name + "\" does not exist")
        log_name_list.append(log_name)

      except Exception as ex:
        print("[main_plot.py]", end=" ")
        print(ex)

  except Exception as ex:
    print("[main_plot.py]", end=" ")
    print(ex)



  try:
    loss_list = []
    val_loss_list = []
    success_list = []
    type_list = []

    for log_name in log_name_list:
      loss, val_loss, success, type = read_file(log_name)
      loss_list.append(loss)
      val_loss_list.append(val_loss)
      success_list.append(success)
      type_list.append(type)

    plot_log(log_name_list, loss_list, val_loss_list, success_list, type_list)

  except Exception as ex:
    print("[main_plot.py]", end=" ")
    print(ex)
