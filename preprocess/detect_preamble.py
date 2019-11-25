import math

import global_vars

num_half_bit = global_vars.num_half_bit
num_bit = 2 * num_half_bit
num_bit_preamble = global_vars.bit_preamble * num_bit
num_bit_extra = global_vars.bit_extra * num_bit




def detect_preamble(sample):
  # preamble mask
  mask = [1.0] * 2 * num_half_bit  # 1
  mask += [-1.0] * num_half_bit  # 2
  mask += [1.0] * num_half_bit
  mask += [-1.0] * 2 * num_half_bit  # 3
  mask += [1.0] * num_half_bit  # 4
  mask += [-1.0] * num_half_bit
  mask += [-1.0] * 2 * num_half_bit  # 5
  mask += [1.0] * 2 * num_half_bit  # 6

  mask2 = mask[:]
  for idx in range(len(mask2)):
    mask2[idx] *= -1.0

  max_idx = 0
  max_score = 0
  state = 0

  #for idx in range(num_bit_extra):
  for idx in range(50, 100):
    score = 0
    score2 = 0
    for mask_idx in range(len(mask)):
      score += mask[mask_idx] * sample[idx + mask_idx]
      score2 += mask2[mask_idx] * sample[idx + mask_idx]
    if score > max_score:
      max_idx = idx
      max_score = score
      state = -1
    if score2 > max_score:
      max_idx = idx
      max_score = score2
      state = 1

  return max_idx + num_bit_preamble, state
