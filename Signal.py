# -*- coding: utf-8 -*-
import math

import global_vars
from decoding import detect_preamble

num_half_bit = global_vars.num_half_bit
num_bit_data = global_vars.bit_data * 2 * num_half_bit


class Signal:
  file_name = ""
  answer = ""
  answer_samples = list()
  samples = list()
  std_samples = list()
  rev_cut_std_samples = list()
  index = 0;
  level = 0;

  def __init__(self, file_name, data):
    self.file_name = file_name
    self.answer = [int(i) for i in data[0]]
    self.samples = list()
    for idx in range(1, len(data)-1):  # -1은 제일 뒤의 '\n'을 제거해주기 위함
      self.samples.append(data[idx])
    self.samples = [float(i) for i in self.samples]

    # std_samples
    avg = 0
    for sample in self.samples:
      avg += sample
    avg /= len(self.samples)

    std = 0
    for sample in self.samples:
      std += pow(sample - avg, 2)
    std /= len(self.samples)
    std = math.sqrt(std)

    self.std_samples = [((i - avg) / std) for i in self.samples]

    for idx in range(len(self.std_samples)):
      if self.std_samples[idx] > 1:
        self.std_samples[idx] = 1
      if self.std_samples[idx] < -1:
        self.std_samples[idx] = -1

    # detect_preamble & rev_cut_std_samples
    self.index, self.level = detect_preamble(self.std_samples)
    self.rev_cut_std_samples = self.std_samples[self.index : self.index + num_bit_data]
    if self.level == 1:
      for idx in range(len(self.rev_cut_std_samples)):
        self.rev_cut_std_samples[idx] *= -1.0

    # answer_samples
    level = -1
    for bit in self.answer:
      for i in range(0, num_half_bit):
        self.answer_samples.append(level)

      if bit:
        for i in range(0, num_half_bit):
          self.answer_samples.append(level)
      else:
        level *= -1
        for i in range(0, num_half_bit):
          self.answer_samples.append(level)
      level *= -1
