clean_messages = False

def get_node_id(time, total_time = 3600, total_node_id = 20):
    return (time%total_time)%total_node_id

def get_message_id(time, total_time = 3600, total_node_id = 20):
    return (time%total_time)//total_node_id

def binary_to_decimal(binary_array):
    decimals = []

    for binary_row in binary_array:
        decimal_num = 0
        power = 7 
        for bit in binary_row:
            decimal_num += bit * (2 ** power)
            power -= 1
        decimals.append(int(decimal_num))

    return decimals

def decimal_to_binary(decimal_list):
    binary_array = []

    for decimal_num in decimal_list:
        binary_row = [0] * 8
        for i in range(7, -1, -1):
            if decimal_num >= 2**i:
                binary_row[7 - i] = 1
                decimal_num -= 2**i
        binary_array.append(binary_row)

    return binary_array