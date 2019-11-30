data_path = "../data/"
folder_name = "191019"
target_name = "reverse_preamble_std_index"
source_path = data_path + folder_name + "/"
target_path = data_path + folder_name + "_" + target_name + "/"



num_half_bit = 25
bit_preamble = 6
bit_data = 128
bit_extra = 12



file_size = 3000
file_name_list = []


for a in ["100", "200", "300", "400"]:
    for b in ["0", "l100", "r100"]:
        for c in ["0", "45", "90", "135"]:
            file_name_list.append(a + "_" + b + "_" + c)
