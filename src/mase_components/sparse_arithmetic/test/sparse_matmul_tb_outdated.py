#!/usr/bin/env python3

# This script tests the fixed point linear
import os, math, logging
import sys

from mase_cocotb.random_test import *
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.x_width = 8
        self.x_frac_width = 0
        self.y_width = 8
        self.y_frac_width = 0
        # self.bias_width = 16
        # self.bias_frac_width = 0
        self.data_out_width = 8 + 8 + int(math.log2(1*2))  # $clog2(NONSPARSE_BLOCK_NUM * BLOCK_SIZE)
        self.data_out_frac_width = 0
        # self.has_bias = 1

        self.x_rows = 2
        self.x_columns = 8
        self.y_rows = self.x_columns
        self.y_columns = 2
        self.iterations = 5  # not used for simple_matmul
        self.block_num = 4
        self.sparse_block_num = 3
        self.block_size = int(self.x_columns/self.block_num)
        
        self.x = SparseRandomSource(
            name="x",
            samples=samples * self.iterations,
            num=self.x_rows * self.x_columns,
            max_stalls=0,
            debug=debug,
            # sparsity-related configs
            block_num =self.block_num,
            sparse_block_num=self.sparse_block_num,
            block_size=self.block_size
        )
        self.y = RandomSource(
            name="y",
            samples=samples * self.iterations,
            num=self.y_rows * self.y_columns,
            max_stalls=0,
            debug=debug,
            # sparsit-related configs
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
        DEFAULT_CONFIG = {
            "A_TOTAL_DIM0": self.x_columns * self.iterations,  # M
            "A_TOTAL_DIM1": self.x_rows,  # TODO: A_DEPTH_DIM1 != 1 
            "B_TOTAL_DIM0": self.y_columns, # TODO: B_DEPTH_DIM_0 != 1
            "B_TOTAL_DIM1": self.y_rows * self.iterations, 
            "A_COMPUTE_DIM0": self.x_columns,
            "A_COMPUTE_DIM1": self.x_rows,  # N
            "B_COMPUTE_DIM0": self.y_columns,  # K
            "B_COMPUTE_DIM1": self.y_rows,  # M=M; Must equalt to A_COMPUTE_DIM0
            "A_WIDTH": self.x_width,
            "A_FRAC_WIDTH": self.x_frac_width,
            "B_WIDTH": self.y_width,
            "B_FRAC_WIDTH": self.y_frac_width,
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            # Sparsity
            "BLOCK_NUM": self.block_num,
            "SPARSE_BLOCK_NUM": self.sparse_block_num
        }
        
        return DEFAULT_CONFIG


    def sw_compute(self):
        final = []
        ref = []
        for i in range(self.samples):
            acc = [0 for _ in range(self.x_rows * self.y_columns)]
            for w in range(self.x_rows):
                for j in range(self.iterations):
                    id = i * self.iterations + j
                    current_row = [self.x.data[id][w*self.x_columns + i] for i in range(self.x_columns)]
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
            dut.a_ready.value,
            dut.a_valid.value,
            dut.b_ready.value,
            dut.b_valid.value,
            dut.out_ready.value,
            dut.out_valid.value,
        )
    )


@cocotb.test()
async def test_sparse_matmul(dut):
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
    dut.b_valid.value = 0
    dut.a_valid.value = 0
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
        dut.b_valid.value = test_case.y.pre_compute()
        dut.a_valid.value = test_case.x.pre_compute()
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
        dut.b_valid.value, dut.b_data.value = test_case.y.compute(
            dut.b_ready.value
        )
        dut.a_valid.value, dut.a_data.value = test_case.x.compute(
            dut.a_ready.value
        )
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
    check_results(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(
        module_param_list=[tb.get_dut_parameters()],
        extra_build_args=["--unroll-count", "3000"],
        trace=True
    )