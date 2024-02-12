BW = 32

def arith_shift_right(x:str, bias):
    ''' simulates >>> operator in Verilog '''
    if x[0] == '1':
        # negative
        return '1'*bias + x[0: BW-bias]
    else:
        return '0'*bias + x[0: BW-bias]
    
def dec2binstr(x:int):
    if (x >= 0):
        signed = '0'
        tmp = x
    else:
        signed = '1'
        tmp = 2**BW - abs(x)
    
    tmp = bin(tmp).split('b')[-1]  # tmp = '1010'
    tmp = signed*(32-len(tmp)) + tmp  # tmp = '1'*28 + '1010'
    return tmp

    
def main():
    bias = 2
    x_list = [-1, 0, -2, -3, -4, -60] # raw dec input
    x_bin_list = [dec2binstr(x) for x in x_list] # raw bin input
    
    y_sw_list = [int(x/2**bias) for x in x_list]  # sw dec output
    y_sw_bin_list = [dec2binstr(y) for y in y_sw_list]  # hw bin output
    y_hw_bin_list = [arith_shift_right(x, bias) for x in x_bin_list]  # hw bin output
    
    print('x_list: ', x_list)
    print('x_bin_list: ', x_bin_list)
    print('y_sw_list: ', y_sw_list)
    print('y_sw_bin_list: \t', y_sw_bin_list)
    print('y_hw_bin_list: \t', y_hw_bin_list)
    



if __name__ == '__main__':
    main()