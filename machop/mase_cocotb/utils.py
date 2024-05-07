import random
from copy import copy
import itertools

from cocotb.triggers import RisingEdge
import torch
from torch import Tensor

from mase_cocotb.z_qlayers import quantize_to_int

from functools import partial
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer


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

    # Quantize
    quantizer = partial(integer_quantizer, **q_config)
    q_tensor = quantizer(tensor)

    # Convert to integer format
    q_tensor = (q_tensor * 2 ** q_config["frac_width"]).int()

    # Split into chunks according to parallelism in each dimension
    # parallelism[0]: along rows, parallelism[1]: along columns
    dim_0_split = q_tensor.split(parallelism[0], dim=0)
    dim_1_split = [x.split(parallelism[1], dim=1) for x in dim_0_split]
    blocks = []
    # Flatten the list of blocks
    for i in range(len(dim_1_split)):
        for j in range(len(dim_1_split[i])):
            blocks.append(dim_1_split[i][j].flatten().tolist())
    breakpoint()
    return blocks
