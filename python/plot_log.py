import matplotlib.pyplot as plt
import numpy as np

def plot_log(log_name_list, loss_list, val_loss_list, success_list):
  try:
    model_size = len(log_name_list)

    epoch_list = []
    for loss in loss_list:
      epoch_list.append(len(loss))



    plt.subplot(121)

    for idx in range(model_size):
      val_loss = val_loss_list[idx]
      epoch = epoch_list[idx]
      plt.plot(np.append(np.roll(val_loss, 1), val_loss[epoch-1]))

    plt.title("validation loss")
    plt.grid(which="major", axis="y", linestyle="solid")
    plt.grid(which="minor", axis="y", linestyle="dashed")
    plt.legend(log_name_list)
    plt.xlabel("epoch")
    plt.xlim(1, max(epoch_list)+1)
    plt.yscale("log")

    if model_size == 1:
      loss = loss_list[0]
      val_loss = val_loss_list[0]
      epoch = epoch_list[0]
      plt.plot(np.append(np.roll(loss, 1), loss[epoch-1]))
      plt.text(epoch, loss[epoch-1], str(round(loss[epoch-1], 2)), ha="center", va="bottom")
      plt.text(epoch, val_loss[epoch-1], str(round(val_loss[epoch-1], 2)), ha="center", va="bottom")
      plt.legend(["val_loss", "loss"])

    plt.ylim(top=1)



    plt.subplot(122)

    for idx in range(model_size):
      success = success_list[idx]
      plt.bar(idx+1, success, width=0.5)
      plt.text(idx+1, success, str(round(success, 2)), ha="center", va="bottom")

    plt.title("success rate")
    plt.grid(which="major", axis="y", linestyle="solid")
    plt.grid(which="minor", axis="y", linestyle="dashed")
    plt.xlabel("model")
    plt.xticks(np.arange(1, model_size+1), log_name_list)
    plt.xlim(0, model_size+1)
    plt.ylim(0, 1)

    plt.show()

  except Exception as ex:
    print("[plot_log.py]", end=" ")
    print(ex)
