def binary_int(num, num_bits=8):
    code = bin(num)[2:]
    if len(code) > num_bits:
        print("ERROR: num needs more bits")
        return '0' * num_bits
    else:
        return '0' * (num_bits-len(code)) + code


def int_from_bin(code):
    return int(code, 2)


# Gives the Fibonacci-valid int number that is the closest (down) to our int number 'num'
def fib_code_int_down(num, num_bits=8):
    code = list(binary_int(num, num_bits))
    count = 0
    for i in range(len(code)):
        if code[i] == '1':
            count += 1
            if count >= 2:
                code[i] = '0' # Remove the problem
                # Then we need to make the number as big as possible by placing only 10101010101... until the end of the floating point representation
                one_next = True
                for j in range(i+1, len(code)):
                    if one_next:
                        code[j] = '1'
                        one_next = False
                    else:
                        code[j] = '0'
                        one_next = True
                break
        else:
            count = 0

    code = ''.join(code)
    return int_from_bin(code)


# Gives the Fibonacci-valid int number that is the closest (up) to our int number 'num'
def fib_code_int_up(num, num_bits=8):
    code = list(binary_int(num, num_bits))
    count = 0
    for i in range(len(code)):
        if code[i] == '1':
            count += 1
            if count >= 2:

                one_next = True
                for j in range(i - 2, -1, -1):
                    if one_next:
                        code[j] = '1'
                        one_next = False
                    else:
                        code[j] = '0'
                        one_next = True
                    if j > 0 and code[j - 1] == '0':
                        break

                i -= 1
                while i < len(code):
                    code[i] = '0'  # Remove the problem (and all subsequent problems on the right)
                    i += 1
                break

        else:
            count = 0

    code = ''.join(code)
    return int_from_bin(code)


# Gives the best Fibonacci-valid int approximation for 'num'
def fib_code_int(num, num_bits=8):
    if num <= 0:
        return num

    down = fib_code_int_down(num, num_bits)
    up = fib_code_int_up(num, num_bits)
    dist_down = abs(num - down)
    dist_up = abs(num - up)
    if dist_down < dist_up:
        return down
    else:
        return up

