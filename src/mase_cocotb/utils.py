import random
from copy import copy
import itertools

from cocotb.triggers import RisingEdge
import torch
from torch import Tensor
import sys

sys.path.append("../")
from mase_cocotb.z_qlayers import quantize_to_int

from functools import partial
from chop.nn.quantizers import integer_quantizer


# Apparently this function only exists in Python 3.12 ...
def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


# Apparently this function only exists in Python 3.12 ...
def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def binary_encode(x):
    assert x in [-1, 1]
    return 0 if x == -1 else 1


def binary_decode(x):
    assert x in [0, 1]
    return -1 if x == 0 else 1


async def bit_driver(signal, clk, prob):
    while True:
        await RisingEdge(clk)
        signal.value = 1 if random.random() < prob else 0


def sign_extend_t(value: Tensor, bits: int):
    sign_bit = 1 << (bits - 1)
    return (value.int() & (sign_bit - 1)) - (value.int() & sign_bit)


def sign_extend(value: int, bits: int):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def signed_to_unsigned(value: Tensor, bits: int):
    mask = (1 << bits) - 1
    return value & mask


def floor_rounding(value, in_frac_width, out_frac_width):
    if in_frac_width > out_frac_width:
        return value >> (in_frac_width - out_frac_width)
    elif in_frac_width < out_frac_width:
        return value << (in_frac_width - out_frac_width)
    return value


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def int_floor_quantizer(x: Tensor, width: int, frac_width: int, signed=True):
    if signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    scale = 2**frac_width
    return torch.clamp(torch.floor(x.mul(scale)), int_min, int_max).div(scale)


def random_2d_dimensions():
    compute_dim0 = random.randint(2, 3)
    compute_dim1 = random.randint(2, 3)
    total_dim0 = compute_dim0 * random.randint(1, 3)
    total_dim1 = compute_dim1 * random.randint(1, 3)
    return compute_dim0, compute_dim1, total_dim0, total_dim1


def verilator_str_param(s):
    return f'"{s}"'


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def fixed_preprocess_tensor(tensor: Tensor, q_config: dict, parallelism: list) -> list:
    """Preprocess a tensor before driving it into the DUT.
    1. Quantize to requested fixed-point precision.
    2. Convert to integer format to be compatible with Cocotb drivers.
    3. Split into blocks according to parallelism in each dimension.

    Args:
        tensor (Tensor): Input tensor
        q_config (dict): Quantization configuration.
        parallelism (list): Parallelism in each dimension.

    Returns:
        list: Processed blocks in nested list format.
    """
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)

    if len(parallelism) == 1:
        parallelism = [1, parallelism[0]]

    # * Flatten batch dimension
    tensor = tensor.view((-1, tensor.shape[-1]))

    # Quantize
    quantizer = partial(integer_quantizer, **q_config)
    q_tensor = quantizer(tensor)

    # Convert to integer format
    q_tensor = (q_tensor * 2 ** q_config["frac_width"]).int()
    q_tensor = signed_to_unsigned(q_tensor, bits=q_config["width"])

    # Split into chunks according to parallelism in each dimension
    # parallelism[0]: along rows, parallelism[1]: along columns
    dim_0_split = q_tensor.split(parallelism[0], dim=0)
    dim_1_split = [x.split(parallelism[1], dim=1) for x in dim_0_split]
    blocks = []
    # Flatten the list of blocks
    for i in range(len(dim_1_split)):
        for j in range(len(dim_1_split[i])):
            blocks.append(dim_1_split[i][j].flatten().tolist())

    return blocks


def large_num_generator(large_num_thres=127, large_num_limit=500, large_num_prob=0.1):
    """
    Generator large numbers & small numbers with a given probability distribution.
    Default: 500 >= abs(large number) >= 128
    """
    if random.random() < large_num_prob:
        if random.random() < 0.5:
            return random.randint(large_num_thres + 1, large_num_limit)
        else:
            return random.randint(-large_num_limit, -(large_num_thres + 1))
    else:
        return random.randint(-large_num_thres, large_num_thres)


def fixed_cast(val, in_width, in_frac_width, out_width, out_frac_width):
    if in_frac_width > out_frac_width:
        val = val >> (in_frac_width - out_frac_width)
    else:
        val = val << (out_frac_width - in_frac_width)
    in_int_width = in_width - in_frac_width
    out_int_width = out_width - out_frac_width
    if in_int_width > out_int_width:
        if val >> (in_frac_width + out_int_width) > 0:  # positive value overflow
            val = 1 << out_width - 1
        elif val >> (in_frac_width + out_int_width) < -1:  # negative value overflow
            val = -(1 << out_width - 1)
        else:
            val = val
            # val = int(val % (1 << out_width))
    return val  # << out_frac_width  # treat data<out_width, out_frac_width> as data<out_width, 0>



def random_gen_sparse(sparsity=0.8):
    if random.random() < sparsity:
        return 0
    else:
        return 1
    
def random_gen_block_sparse(block_size, block_num, sparse_block_num, num) -> list:
    # naming conventions:
    # the vector to return contains num elements
    # these elements are grouped as several processing units
    # each unit consists of `block_num` blocks, with each block containing `block_size` elements
    result = []
    
    unit_num = num/(block_num*block_size)
    assert int(unit_num) == unit_num
    unit_num = int(unit_num)
    
    for i in range(unit_num):  # for each processing unit
        sparse_block_ids = random.sample(list(range(block_num)), sparse_block_num)
        # sparse_block_ids = [0, 1]
        for block_id in range(block_num):
            if block_id in sparse_block_ids:
                result += [0]*block_size  # add an all-zero block
            else:
                result += [random.randint(0, 30) for k in range(block_size)]  # add an non-zero block
                
    return result


def printMat(tensor, dim0, dim1):
    # dim0 for row size, dim1 for col size
    assert len(tensor) == dim0*dim1, "Incorrect tensor dimension"
    
    print(' ', end='\t')
    for j in range(dim1):
        print('(%s)'%str(j), end='\t')
    
    print()
    
    for i in range(dim0):
        print('(%s)'%str(i), end='\t')
        for j in range(dim1):
            e = tensor[i*dim1 + j]
            print(e, end='\t')
        print()
    
def sparse2COO(sparse_tensor_flattened, dim0, dim1, align_size=None):
    ''' Convert sparse matrix to COO format.
        align_size: all dense rows must be padded with zero to have their size aligned.
    '''
    val = []
    row_table = []
    col_table = []
    
    assert len(sparse_tensor_flattened) == dim0*dim1, "Unmatched sparse tensor dimension"
    
    for i in range(dim0):
        for j in range(dim1):
            ind = i*dim1 + j
            e = sparse_tensor_flattened[ind]
            if (e != 0):
                val.append(e)
                row_table.append(i)
                col_table.append(j)
    
    if align_size != None and len(val) < align_size:
        pad = align_size - len(val)
        val += [-1]*pad
        row_table += [-1]*pad
        col_table += [-1]*pad
    return val, row_table, col_table

def COO2Sparse(coo_val, coo_row, coo_col, dim0, dim1):
    assert len(coo_val) == len(coo_row), "Unmatched COO tuple."
    assert len(coo_row) == len(coo_col), "Unmathced COO tuple."
    
    sparse_tensor_flattened = [0 for i in range(dim0*dim1)]
    for i in range(len(coo_val)):
        val = coo_val[i]
        row = coo_row[i]
        col = coo_col[i]
        if (row == -1 or col == -1):
            continue
        ind = row*dim1 + col
        sparse_tensor_flattened[ind] = val
                    
    return sparse_tensor_flattened


def sparse2CSR(sparse_tensor_flattened, dim0, dim1, align_size=None):
    ''' Convert sparse matrix to COO format.
        align_size: all dense rows must be padded with zero to have their size aligned.
    '''
    val = []
    col_index = []
    row_bound = []
    
    assert len(sparse_tensor_flattened) == dim0*dim1, "Unmatched sparse tensor dimension"
    
    count = 0 
    row_bound.append(count)           
    for i in range(dim0):
        for j in range(dim1):
            ind = i*dim1 + j
            e = sparse_tensor_flattened[ind]
            if (e != 0):
                val.append(e)
                col_index.append(j)
                count += 1
        row_bound.append(count)
    
    if align_size != None:
        if (len(val) < align_size):
            pad = align_size - len(val)
            val += [0]*pad
            col_index += [-1]*pad
        elif (len(val) > align_size):
            print('WARNING: actual number of non-zero elements larger than align_size')
    if (len(row_bound) == 4):
        print('#######Catch it!')
        print(sparse_tensor_flattened)
        print(val)
        print(col_index)
        print(row_bound)
    return val, col_index, row_bound

def CSR2Sparse(csr_val, csr_index, csr_bound, dim0, dim1):
    assert len(csr_val) == len(csr_index), "Unmatched CSR tuple."
    assert len(csr_bound) == dim0 + 1, "Unmathced CSR tuple."
    
    sparse_tensor_flattened = [0 for i in range(dim0*dim1)]
    count = 0
    row = 0
    
    for row in range(len(csr_bound)-1):
        while (count < csr_bound[row+1]):  # step to the next non-empty row
            val = csr_val[count]
            col = csr_index[count]
            if (row == -1 or col == -1):
                continue
            else:
                sparse_tensor_flattened[row*dim1 + col] = val
                count += 1
            
    return sparse_tensor_flattened

