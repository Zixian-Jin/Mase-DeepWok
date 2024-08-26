`timescale 1ns / 1ps
/*
 * Module: block_router
 * Description: this module multiplexes IN_BLOCK_NUM input blocks to OUT_BLOCK_NUM output blocks
 *              multiplexing is controlled by non_zero_sel, where there are OUT_BLOCK_NUM bits of 0.
 */
module block_router #(
    parameter IN_BLOCK_NUM = 3,
    parameter BLOCK_SIZE = 4,
    parameter OUT_BLOCK_NUM = 2,
    parameter IN_WIDTH = 16
) (
    input clk,
    input rst,
    input [IN_BLOCK_NUM-1: 0] nonzero_sel,
    input [IN_WIDTH-1 :0] in_data [IN_BLOCK_NUM * BLOCK_SIZE -1 :0],
    input in_valid,
    output in_ready,
    output [IN_WIDTH-1 :0] out_data [OUT_BLOCK_NUM * BLOCK_SIZE-1 :0],
    output out_valid,
    input out_ready
);

    initial begin
        assert (IN_BLOCK_NUM >= OUT_BLOCK_NUM)
            else $fatal("OUT_BLOCK_NUM must be no larger than IN_BLOCK_NUM!");
    end

    
    logic [IN_WIDTH-1 :0] reg_out [OUT_BLOCK_NUM * BLOCK_SIZE-1 :0];
    integer i, j, nonzero_block_id;

    always_comb begin

        for (int k = 0; k < OUT_BLOCK_NUM * BLOCK_SIZE; k++) begin
            reg_out[k] = '0;
        end

        i = 0;
        j = 0;
        nonzero_block_id = 0;
        for (int block_id = 0; block_id < IN_BLOCK_NUM; block_id++) begin
            if (nonzero_sel[block_id] == 1'b0) begin
                // non-zero block
                for (int k=0; k<BLOCK_SIZE; k++) begin
                    i = block_id*BLOCK_SIZE + k;
                    j = nonzero_block_id*BLOCK_SIZE + k;
                    reg_out[j] = in_data[i];
                end
                nonzero_block_id = nonzero_block_id + 1;
            end
        end
    end

    // Cocotb/verilator does not support array flattening, so
    // we need to manually add some reshaping process.

    logic [$bits(reg_out)-1 : 0] reg_out_1d_in;
    logic [$bits(reg_out)-1 : 0] reg_out_1d_out;

    // Casting array for product vector
    for (genvar i = 0; i < OUT_BLOCK_NUM * BLOCK_SIZE; i++) begin : reshape_in
        assign reg_out_1d_in[IN_WIDTH*i+IN_WIDTH-1:IN_WIDTH*i] = reg_out[i];
    end

    skid_buffer #(
        .DATA_WIDTH($bits(reg_out))
    ) register_slice (
        .clk           (clk),
        .rst           (rst),
        .data_in       (reg_out_1d_in),
        .data_in_valid (in_valid),
        .data_in_ready (in_ready),
        .data_out      (reg_out_1d_out),
        .data_out_valid(out_valid),
        .data_out_ready(out_ready)
    );

    // Casting array for product vector
    for (genvar i = 0; i < OUT_BLOCK_NUM * BLOCK_SIZE; i++) begin : reshape_out
        assign out_data[i] = reg_out_1d_out[IN_WIDTH*i+IN_WIDTH-1:IN_WIDTH*i];
    end


endmodule
