class SignalSet:
  def __init__(self):
    self.train = []
    self.validation = []
    self.test = []

  def concatenate(self, src):
    self.train += src.train
    self.validation += src.validation
    self.test += src.test

  def random_train_set(self, random_index):
    train_backup = self.train[:]
    self.train = []

    for idx in range(len(train_backup)):
      self.train.append(train_backup[random_index[idx]])
