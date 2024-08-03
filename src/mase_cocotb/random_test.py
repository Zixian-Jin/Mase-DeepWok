#!/usr/bin/env python3

# This script inlcudes the input and output components that have random behaviours.
# They are used for capturing all the possible dataflow computation behaviours.
import random, os, math, logging, sys
from .utils import *


# A source node that randomly sends out a finite number of
# data using handshake interface
class RandomSource:
    def __init__(
        self,
        samples=10,
        num=1,
        max_stalls=100,
        is_data_vector=True,
        name="",
        data_specify=[],
        debug=False,
        arithmetic=None,
        fix_seed=False,
    ):
        # Use in debug
        if fix_seed:
            random.seed(0)
        assert num > 0, "Invalid num for source {}".format(name)
        self.logger = logging.getLogger(name)
        if debug:
            logger_level = logging.DEBUG
        else:
            logger_level = logging.INFO
        self.logger.setLevel(logger_level)
        self.name = name
        self.num = num
        self.samples = samples
        self.max_stalls = max_stalls
        self.is_data_vector = is_data_vector
        if arithmetic in ["binary"]:
            self.rand_gen = lambda: binary_encode(random.choice([-1, 1]))
        elif arithmetic in ["ternary"]:
            self.rand_gen = lambda: binary_encode(random.randint(-1, 1))
        elif arithmetic in ["llm-fp16-datain"]:
            self.rand_gen = lambda: large_num_generator(
                large_num_thres=127, large_num_limit=500, large_num_prob=0.1
            )
        elif arithmetic in ["llm-fp16-weight"]:
            self.rand_gen = lambda: random.randint(-5, 5)
        else:
            self.rand_gen = lambda: random.randint(0, 30)
            # self.rand_gen = lambda: random.randint(-random.randint(15, 30), random.randint(15, 30))

        if len(data_specify) == 0:
            if is_data_vector:
                self.data = [
                    [self.rand_gen() for _ in range(num)] for _ in range(samples)
                ]
            else:
                self.data = [
                    self.rand_gen() for _ in range(num) for _ in range(samples)
                ]
        else:
            self.data = data_specify

        self.dummy = (
            [self.rand_gen() for _ in range(num)] if is_data_vector else self.rand_gen()
        )

        self.stall_count = 0
        # Buffer the random choice
        self.random_buff = 0

    def pre_compute(self):
        # randomly stops feeding data before reaching the max stalls
        self.random_buff = random.randint(0, 1)
        self.stall_count += self.random_buff
        if (not self.random_buff) or self.stall_count > self.max_stalls:
            return 1
        self.logger.debug(
            "pre_compute: source {} skips an iteration.".format(self.name)
        )
        return 0

    def compute(self, next_ready):
        """The compute simulates the synchronous computation for data"""
        to_feed = (not self.is_empty()) and next_ready
        if self.is_empty():
            data = self.dummy
        else:
            data = self.data[-1]
        if not to_feed:
            self.logger.debug(
                "source {} cannot feed any token because of back pressure.".format(
                    self.name
                )
            )
            return (not self.is_empty()), data
        if (not self.random_buff) or self.stall_count > self.max_stalls:
            data
            self.data.pop()
            self.logger.debug(
                "source {} feeds a token. Current depth = {}/{}".format(
                    self.name, len(self.data), self.samples
                )
            )
            return 1, data
        return 0, data

    def is_empty(self):
        return len(self.data) == 0


# A sink node that randomly absorbs a finite number of
# data using handshake interface
class RandomSink:
    def __init__(self, samples=10, num=1, max_stalls=100, name="", debug=False):
        assert num > 0, "Invalid num for sink {}".format(name)
        self.logger = logging.getLogger(name)
        if debug:
            logger_level = logging.DEBUG
        else:
            logger_level = logging.INFO
        self.logger.setLevel(logger_level)
        self.data = []
        self.name = name
        self.num = num
        self.samples = samples
        self.max_stalls = max_stalls
        self.stall_count = 0
        self.trystall = 0

    def pre_compute(self, prevalid):
        to_absorb = (not self.is_full()) and prevalid
        if not to_absorb:
            self.logger.debug(
                "pre_compute: a sink {} cannot absorb any token because of no valid data.".format(
                    self.name
                )
            )
            return not self.is_full()
        # randomly stops absorbing data before reaching the max stalls
        self.trystall = random.randint(0, 1)
        self.stall_count += self.trystall
        if (not self.trystall) or self.stall_count > self.max_stalls:
            return 1
        self.logger.debug("pre_compute: sink {} skips an iteration.".format(self.name))
        return 0

    def compute(self, prevalid, datain):
        to_absorb = (not self.is_full()) and prevalid
        if not to_absorb:
            return 0
        if (not self.trystall) or self.stall_count > self.max_stalls:
            self.data.append(datain)
            self.logger.debug(
                "sink {} absorbs a token. Current depth = {}/{}".format(
                    self.name, len(self.data), self.samples
                )
            )
            return 1
        return 0

    def is_full(self):
        return len(self.data) == self.samples


class SparseRandomSource(RandomSource):
    def __init__(
        self,
        samples=10,
        num=1,
        dim0=1, # row
        dim1=1, # col
        max_stalls=100,
        is_data_vector=True,
        name="",
        data_specify=[],
        debug=False,
        arithmetic=None,
        fix_seed=False,
        
        # sparsity related configs
        compression=None,  # data compression format
        block_num=3,  # no. blocks per processing unit
        sparse_block_num=2,
        block_size=4  # no. elements per block
    ):
        super().__init__(
            samples,
            num,
            max_stalls,
            is_data_vector,
            name,
            data_specify,
            debug,
            arithmetic,
            fix_seed,
        )
        
        # Overwrite self.data if sparse
        self.rand_gen = lambda: random_gen_block_sparse(block_size, block_num, sparse_block_num, num)
        
        # For switching between different compression formats
        # self.data_pack = {
        #     "COO": {"val":[], "row":[], "col":[]},
        #     "CSR": {"val":[], "col_index":[], "row_bound":[]},
        #     "CSC": {"val":[], "row_index":[], "col_bound":[]}
        # }
        
        # Implementation varies between different compression formats and will be overloaded.
        self.data_pack = {"val":[], "index":[]}
        self.dummy_pack = {"val":[], "index":[]}
                
        if is_data_vector:
            val_sparse = [self.rand_gen() for _ in range(samples)]
            dummy_sparse = [self.rand_gen() for _ in range(num)]
        else:
            # TODO
            pass
        
        # Overwrite self.data and create self.row & self.col if compression
        if compression == 'COO':
            assert dim0*dim1 == num
            # Overwrite self.dummy
            self.dummy = [-1 for _ in range((dim1-sparse_block_num)*dim0)]
            self.row = []
            self.col = []
            for i in range(samples):
                val, row_table, col_table = self.sparse2COO(self.data[i], dim0, dim1, align_size=(dim1-sparse_block_num)*dim0)
                self.data[i] = val
                self.row.append(row_table)
                self.col.append(col_table)
        elif compression == 'CSR':
            assert dim0*dim1 == num
            # Overload self.data_pack
            self.data_pack = {"val":[], "col_index":[], "row_bound":[]}
            # Overload self.dummy_pack
            self.dummy_pack['val'] = [0 for _ in range((dim1-sparse_block_num)*dim0)]
            self.dummy_pack['col_index'] = [-1 for _ in range((dim1-sparse_block_num)*dim0)]
            self.dummy_pack['row_bound'] = [0 for _ in range(dim0+1)]
            
            csr_val_dense = []
            csr_col_index = []
            csr_row_bound = []
            for i in range(samples):
                csr_val_dense, csr_col_index, csr_row_bound = sparse2CSR(val_sparse[i], dim0, dim1, align_size=(dim1-sparse_block_num)*dim0)
                self.data_pack["val"].append(csr_val_dense)
                self.data_pack["col_index"].append(csr_col_index)
                self.data_pack["row_bound"].append(csr_row_bound)
                

        elif compression == 'CSC':
            pass
        else:
            pass
        
    def compute(self, next_ready):
        """The compute simulates the synchronous computation for data"""
        to_feed = (not self.is_empty()) and next_ready
        data = {}   # implementation varies between different compression formats
        if self.is_empty():
            data = self.dummy_pack
        else:
            for (k, v) in self.data_pack.items():
                data[k] = v[-1]                
        if not to_feed:
            self.logger.debug(
                "source {} cannot feed any token because of back pressure.".format(
                    self.name
                )
            )
            return (not self.is_empty()), data
        if (not self.random_buff) or self.stall_count > self.max_stalls:
            data
            for (k, v) in self.data_pack.items():
                v.pop()  
            self.logger.debug(
                "source {} feeds a token. Current depth = {}/{}".format(
                    self.name, len(self.data), self.samples
                )
            )
            return 1, data
        return 0, data

    def is_empty(self):
        return len(self.data_pack['val']) == 0



def check_results(hw_out, sw_out, thres=1):
    assert len(hw_out) == len(
        sw_out
    ), "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))

    if type(hw_out[0]) == list:
        for i in range(len(hw_out)):
            assert (
                # hw_out[i] == sw_out[i]
                compare_lists_approx(hw_out[i], sw_out[i], thres)
            ), "Mismatched output value {}: {} expected = {}".format(
                i, [int(t) for t in hw_out[i]], [int(t) for t in sw_out[i]]
            )
        return True
    else:
        for i in range(len(hw_out)):
            assert (
                # int(hw_out[i]) == int(sw_out[i])
                compare_numbers_approx(int(hw_out[i]), int(sw_out[i]), thres)
            ), "Mismatched output value {}: {} expected = {}".format(
                i, int(hw_out[i]), int(sw_out[i])
            )


def check_results_signed(hw_out, sw_out, thres=1):
    assert len(hw_out) == len(
        sw_out
    ), "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))
    if type(hw_out[0]) == list:
        for i in range(len(hw_out)):
            assert (
                # [i.signed_integer for i in hw_out[i]] == sw_out[i]
                compare_lists_approx(
                    [i.signed_integer for i in hw_out[i]], sw_out[i], thres
                )
            ), "Mismatched output value {}: {} expected = {}".format(
                i,
                [int(t.signed_integer) for t in hw_out[i]],
                [int(t) for t in sw_out[i]],
            )
        return True
    else:
        for i in range(len(hw_out)):
            assert (
                # hw_out[i].signed_integer == int(sw_out[i])
                compare_numbers_approx(hw_out[i].signed_integer, int(sw_out[i]), thres)
            ), "Mismatched output value {}: {} expected = {}".format(
                i, int(hw_out[i].signed_integer), int(sw_out[i])
            )


def analyse_results_signed(hw_out, sw_out, thres=1):
    assert len(hw_out) == len(
        sw_out
    ), "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))
    if type(hw_out[0]) == list:
        error_list = []
        rel_error_list = []
        count = 0
        for i in range(len(hw_out)):
            hw_result = [i.signed_integer for i in hw_out[i]]
            sw_result = sw_out[i]
            # find maximum error of current output vector
            errors = [(hw_result[i] - sw_result[i]) for i in range(len(sw_result))]
            errors = [abs(e) for e in errors]
            error = max(errors)
            try:
                rel_error = abs(error / max([abs(e) for e in sw_result])) * 100
            except:
                # to prevent divide-by-zero
                rel_error = 0

            # append error and rel. error
            error_list.append(error)
            rel_error_list.append(rel_error)
            if error > thres:
                count += 1
        max_error = max(error_list)
        max_error_ind = error_list.index(max_error)
        print("\n--------------------- Error Analysis --------------------")
        print("Sample Num=%d" % len(sw_out))
        print("No. Samples above Error Thres(%d)=%d" % (thres, count))

        print(
            "Absolute Error: max=%d, avg=%d"
            % (max(error_list), (sum(error_list) / len(error_list)))
        )
        print(
            "Relative Error: max={:.2f}%, avg={:.2f}%".format(
                max(rel_error_list), (sum(rel_error_list) / len(error_list))
            )
        )

        # print("where: hw_out={}, sw_out={}".format([i.signed_integer for i in hw_out[max_error_ind]], sw_out[max_error_ind]))
        # print("error_list={}".format(error_list))
        print("--------------------- End of Error Analysis --------------------\n")
    else:  # TODO
        print("N.A.")
        return
        for i in range(len(hw_out)):
            assert (
                # hw_out[i].signed_integer == int(sw_out[i])
                compare_numbers_approx(hw_out[i].signed_integer, int(sw_out[i]), thres)
            ), "Mismatched output value {}: {} expected = {}".format(
                i, int(hw_out[i].signed_integer), int(sw_out[i])
            )


def compare_lists_approx(l1, l2, thres):
    for i in range(len(l1)):
        if abs(l1[i] - l2[i]) > thres:
            return False
    return True


def compare_numbers_approx(n1, n2, thres):
    return abs(n1 - n2) <= thres
