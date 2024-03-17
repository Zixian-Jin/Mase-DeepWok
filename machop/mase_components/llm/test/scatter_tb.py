#!/usr/bin/env python3


# This script tests the fixed point linear

# Manually add mase_cocotb to system path
import sys, os
try:
    p = os.getenv("MASE_RTL")
    assert p != None
except:
    p = os.getenv("mase_rtl")
    assert p != None
p = os.path.join(p, '../')
sys.path.append(p)
###############################################
import os, math, logging

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
        self.data_in_width = 16
        self.in_rows = 20
        self.in_columns = 4
        
        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=self.in_rows * self.in_columns,
            max_stalls=0,
            debug=debug,
            arithmetic="llm-fp16"
        )
        
        self.outputs = RandomSink(samples=samples, max_stalls=0, debug=debug)
        
        self.samples = samples
        self.ref = self.sw_compute()
        # self.ref = self.sw_cast(
        #     inputs=self.ref,
        #     in_width=self.data_in_width
        #     + self.weight_width
        #     + math.ceil(math.log2(self.iterations * self.in_columns))
        #     + self.has_bias,
        #     in_frac_width=self.data_in_frac_width + self.weight_frac_width,
        #     out_width=self.data_out_width,
        #     out_frac_width=self.data_out_frac_width,
        # )

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "IN_PARALLELISM": self.in_rows,
            "IN_SIZE": self.in_columns,
        }

    def sw_compute(self):
        # for small_large_out only
        final = []
        ref = []
        for i in range(len(self.data_in.data)):
            current_vector = [0]*len(self.data_in.data[0])
            for j in range(len(current_vector)):
                entry = self.data_in.data[i][j]
                if self.sw_large_number_checker(entry, pos=13):
                    # entries with large numbers are masked
                    current_vector[j] = 0
                else:
                    current_vector[j] = entry
            ref.append(current_vector)
        ref.reverse()
        return ref

    def sw_large_number_checker(self, data, pos=14):
        # MSB checker for fixed-point 16
        # data is a signed integer
        if (data > 0):
            return (data >= (2**pos))
        else:
            return (abs(data) >= (2**pos + 1))
        
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
        "{} State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})".format(
            state,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )

def unsignedTosigned(x):
    if x < 2**15:
        return x
    else:
        return x - 2**16
@cocotb.test()
async def test_scatter(dut):
    """Test integer based vector mult"""
    samples = 10
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
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 10):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        debug_state(dut, "Pre-clk")
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out_small.value
        )
        # breakpoint()
        debug_state(dut, "Pre-clk")
        if (
            test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results_signed(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(
        module_param_list=[tb.get_dut_parameters()],
        extra_build_args=["--unroll-count", "3000"],
    )
