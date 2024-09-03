`timescale 1ns/1ps


module element_router #(
    parameter IN_SIZE = 3,
    parameter OUT_SIZE = 2,
    parameter IN_WIDTH = 16
) (
    input clk,
    input rst,
    input [IN_SIZE-1: 0] nonzero_sel,
    input [IN_WIDTH-1 :0] in_data [IN_SIZE-1 :0],
    input in_valid,
    output in_ready,
    output [IN_WIDTH-1 :0] out_data [OUT_SIZE-1 :0],
    output out_valid,
    input out_ready
);


    block_router #(
        .IN_BLOCK_NUM (IN_SIZE),
        .BLOCK_SIZE (1),
        .OUT_BLOCK_NUM (OUT_SIZE),
        .IN_WIDTH (IN_WIDTH)
    ) pseudo_block_router (
        .clk (clk),
        .rst (rst),
        .nonzero_sel (non_zero_sel),
        .in_data (in_data),
        .in_valid (in_valid),
        .in_ready (in_ready),
        .out_data (out_data),
        .out_valid (out_valid),
        .out_ready (out_ready)
    );

endmodule