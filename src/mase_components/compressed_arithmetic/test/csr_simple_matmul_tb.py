#!/usr/bin/env python3

# This script tests the fixed point linear
import os, math, logging
import sys
sys.path.append('/home/zixian/mase-tools/machop')
# sys.path.append('../../../')
###############################################
from mase_cocotb.random_test import *
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import *

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
import copy
debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.x_width = 32
        self.x_frac_width = 0
        self.y_width = 16
        self.y_frac_width = 0
        # self.bias_width = 16
        # self.bias_frac_width = 0
        self.data_out_width = 32
        self.data_out_frac_width = 0
        # self.has_bias = 1

        self.x_rows = 4
        self.x_columns = 15
        self.y_rows = self.x_columns
        self.y_columns = 5
        self.iterations = 1  # not used for simple_matmul
        assert self.iterations == 1, "DEPTH is always 1 for simple_matmul!"
        self.x_row_sparsity = 0.8  # sparsity in each row of x
        self.x_row_sparse_size = int(self.x_row_sparsity * self.x_columns)
        self.x_row_dense_size = self.x_columns - self.x_row_sparse_size

        # Alghouth the compressed_simple_matmul module utilizes element-wise
        # instead of block-wise sparsity, we still keep the block-wise sparse generator syntax
        # as far as code consistency is concerned.  
        self.x = CompressedRandomSource(
            name="x",
            samples=samples * self.iterations,
            num=self.x_rows * self.x_columns,
            dim0 = self.x_rows,
            dim1 = self.x_columns,
            max_stalls=0,
            debug=debug,
            
            # sparsity-related configs
            block_num = self.x_columns,  # granularity=1, element-wise sparsity
            block_size=1,
            sparse_block_num=self.x_row_sparse_size,
            compression="CSR"
        )
        self.y = RandomSource(
            name="y",
            samples=samples * self.iterations,
            num=self.y_rows * self.y_columns,
            max_stalls=0,
            debug=debug,
        )
        # self.bias = RandomSource(
        #     name="bias",
        #     samples=samples,
        #     num=self.in_rows * self.y_columns,
        #     max_stalls=0,
        #     debug=debug,
        # )
        self.outputs = RandomSink(samples=samples, max_stalls=0, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()
        self.ref = self.sw_cast(
            inputs=self.ref,
            in_width=self.x_width
            + self.y_width
            + math.ceil(math.log2(self.iterations * self.x_columns)),
            # + self.has_bias,
            in_frac_width=self.x_frac_width + self.y_frac_width,
            out_width=self.data_out_width,
            out_frac_width=self.data_out_frac_width,
        )

    def get_dut_parameters(self):
        return {
            "X_WIDTH": self.x_width,
            "X_FRAC_WIDTH": self.x_frac_width,
            "Y_WIDTH": self.y_width,
            "Y_FRAC_WIDTH": self.y_frac_width,
            # "BIAS_WIDTH": self.bias_width,
            # "BIAS_FRAC_WIDTH": self.bias_frac_width,
            # "HAS_BIAS": self.has_bias,
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "N": self.x_rows,
            "M": self.x_columns,
            "K": self.y_columns,
            "NZN": self.x_row_dense_size*self.x_rows,  # total dense elements in x
            "NZN_ROW": self.x_row_dense_size
            # "IN_DEPTH": self.iterations,
        }

    def sw_compute(self):
        final = []
        ref = []
        
        # convert X from COO compressed format to sparse format
        x_sparse = []
        for i in range(self.samples):
            sparse_flattened = CSR2Sparse(
                self.x.data_pack['val'][i], 
                self.x.data_pack['col_index'][i], 
                self.x.data_pack['row_bound'][i], 
                self.x_rows, 
                self.x_columns
            )
            x_sparse.append(sparse_flattened)
            
        global sparse_mat_x, sparse_mat_y
        sparse_mat_x = copy.deepcopy(x_sparse[-1])
        sparse_mat_y = copy.deepcopy(self.y.data[-1])
        for id in range(self.samples):
            acc = [0 for _ in range(self.x_rows * self.y_columns)]
            for w in range(self.x_rows):
                current_row = [x_sparse[id][w*self.x_columns + i] for i in range(self.x_columns)]
                for k in range(self.y_columns):
                    current_col = [self.y.data[id][j*self.y_columns + k] for j in range(self.y_rows)]
                    s = [current_row[h]*current_col[h] for h in range(self.x_columns)]
                    acc[w * self.y_columns + k] += sum(s)
            # if self.has_bias:
            #     for j in range(self.x_rows * self.y_columns):
            #         acc[j] += self.bias.data[i][j] << (
            #             self.y_frac_width
            #             + self.x_frac_width
            #             - self.bias_frac_width
            #         )
            ref.append(acc)
        ref.reverse()
        return ref

    def sw_cast(self, inputs, in_width, in_frac_width, out_width, out_frac_width):
        outputs = []
        for j in range(len(inputs)):
            in_list = inputs[j]
            out_list = []
            for i in range(0, len(in_list)):
                in_value = in_list[i]
                if in_frac_width > out_frac_width:
                    in_value = in_value >> (in_frac_width - out_frac_width)
                else:
                    in_value = in_value << (out_frac_width - in_frac_width)
                in_int_width = in_width - in_frac_width
                out_int_width = out_width - out_frac_width
                if in_int_width > out_int_width:
                    if in_value >> (in_frac_width + out_int_width) > 0:
                        in_value = 1 << out_width - 1
                    elif in_value >> (in_frac_width + out_int_width) < 0:
                        in_value = -(1 << out_width - 1)
                    else:
                        in_value = int(in_value % (1 << out_width))
                out_list.append(in_value)
            outputs.append(out_list)
        return outputs


def debug_state(dut, state):
    logger.debug(
        "{} State: (w_ready,w_valid,in_ready,in_valid,out_ready,out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.x_ready.value,
            dut.x_valid.value,
            dut.y_ready.value,
            dut.y_valid.value,
            dut.out_ready.value,
            dut.out_valid.value,
        )
    )


@cocotb.test()
async def test_csr_simple_matmul(dut):
    """Test integer based vector mult"""
    samples = 100
    test_case = VerificationCase(samples=samples)

    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    # Synchronize with the clock
    dut.y_valid.value = 0
    dut.x_valid.value = 0
    dut.out_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 50):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        # dut.bias_valid.value = test_case.bias.pre_compute()
        dut.y_valid.value = test_case.y.pre_compute()
        dut.x_valid.value = test_case.x.pre_compute()
        await Timer(1, units="ns")
        dut.out_ready.value = test_case.outputs.pre_compute(
            dut.out_valid.value
        )
        debug_state(dut, "Pre-clk")
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")
        # dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
        #     dut.bias_ready.value
        # )
        dut.y_valid.value, dut.y_data.value = test_case.y.compute(dut.y_ready.value)
        
        x_valid, csr_data_pack = test_case.x.compute(dut.x_ready.value)
        dut.x_valid.value, dut.x_data.value, dut.x_col_index, dut.x_row_bound = \
            x_valid, csr_data_pack['val'], csr_data_pack['col_index'], csr_data_pack['row_bound']
            # manually reverse the input data such that it is in little endian format
        await Timer(1, units="ns")
        dut.out_ready.value = test_case.outputs.compute(
            dut.out_valid.value, dut.out_data.value
        )
        # breakpoint()
        debug_state(dut, "Pre-clk")
        if (
            test_case.y.is_empty()
            and test_case.x.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"
    print("MAT_X:")
    printMat(sparse_mat_x, test_case.x_rows, test_case.x_columns)
    print("MAT_Y:")
    printMat(sparse_mat_y, test_case.y_rows, test_case.y_columns)
    check_results(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    sparse_mat_x = []
    sparse_mat_y = []
    tb = VerificationCase()
    mase_runner(
        module_param_list=[tb.get_dut_parameters()],
        extra_build_args=["--unroll-count", "3000"],
    )