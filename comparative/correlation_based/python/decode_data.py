from detect_preamble import detect_preamble

import global_vars
num_half_bit = global_vars.num_half_bit
num_bit = 2 * num_half_bit
bit_data = global_vars.bit_data
num_bit_data = global_vars.bit_data * num_bit

mask0L = [1.0] * num_half_bit
mask0L += [-1.0] * num_half_bit
mask0L += [1.0] * num_half_bit
mask0L += [-1.0] * num_half_bit

mask0H = [-1.0] * num_half_bit
mask0H += [1.0] * num_half_bit
mask0H += [-1.0] * num_half_bit
mask0H += [1.0] * num_half_bit

mask1L = [1.0] * num_half_bit
mask1L += [-1.0] * 2 * num_half_bit
mask1L += [1.0] * num_half_bit

mask1H = [-1.0] * num_half_bit
mask1H += [1.0] * 2 * num_half_bit
mask1H += [-1.0] * num_half_bit



def decode_data(sample):
  try:
    start, state = detect_preamble(sample)

    # window with: 1
    shift_range = [i for i in range(0, 1)]

    decoded_bit = []
    for bit in range(bit_data):
      if state == 1:
        mask0 = mask0H
        mask1 = mask1H
      else:
        mask0 = mask0L
        mask1 = mask1L

      max_score = -987654321
      max_value = -1
      cur_shift = 0

      for shift in shift_range:
        score0 = 0
        score1 = 0

        for mask_idx in range(2 * num_bit):
          idx = start - num_half_bit + mask_idx + shift
          if idx >= len(sample): continue
          score0 += mask0[mask_idx] * sample[idx]
          score1 += mask1[mask_idx] * sample[idx]

        if score0 > max_score:
          max_score = score0
          max_value = 0
          cur_shift = shift

        if score1 > max_score:
          max_score = score1
          max_value = 1
          cur_shift = shift

      if max_value == 1: state *= -1
      decoded_bit.append(max_value)
      start += (num_bit + cur_shift)

    return decoded_bit

  except Exception as ex:
    print("[decode_data.py]", end=" ")
    print(ex)
