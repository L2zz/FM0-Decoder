from global_vars import *


num_bit = 2 * num_half_bit
num_bit_data = bit_data * num_bit
num_bit_preamble = bit_preamble * num_bit
num_bit_extra = bit_extra * num_bit

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


def detect_preamble(sample):
    try:
        max_idx = 0
        max_score = 0
        state = 0

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

    except Exception as ex:
        print("[detect_preamble.py]", end=" ")
        print(ex)


def decode_data(sample):
    try:
        start, state = detect_preamble(sample)

        # window with: 3
        shift_range = [i for i in range(-1, 2)]

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
                    if idx >= len(sample):
                        continue
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

            if max_value == 1:
                state *= -1
            decoded_bit.append(max_value)
            start += (num_bit + cur_shift)

        return decoded_bit

    except Exception as ex:
        print("[decode_data.py]", end=" ")
        print(ex)
