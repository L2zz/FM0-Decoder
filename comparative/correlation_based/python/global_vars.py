import datetime
execute_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
data_path = "../../data/191019/"
log_path = "log/" + execute_time



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
