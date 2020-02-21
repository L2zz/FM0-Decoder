import numpy as np

tot_size = 500
test_ratio = 0.2
val_ratio = 0.2

test_size = int(tot_size * test_ratio)
val_size = int(int(tot_size-test_size) * val_ratio)
train_size = tot_size - test_size - val_size



if __name__ == "__main__":
  try:
    tot = np.arange(tot_size)
    np.random.shuffle(tot)

    file = open("../data/random_index/RN_500_1", "w")

    # train set
    for idx in range(train_size):
      file.write(str(tot[idx]) + " ")
    file.write("\n")

    # validation set
    for idx in range(train_size, train_size+val_size):
      file.write(str(tot[idx]) + " ")
    file.write("\n")

    # test set
    for idx in range(train_size+val_size, train_size+val_size+test_size):
      file.write(str(tot[idx]) + " ")
    file.write("\n")

    file.close()

  except Exception as ex:
    print(ex)
