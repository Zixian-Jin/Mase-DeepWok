`timescale 1ns / 1ps
/* 
 * Module: nzc
 * Description: parallel non-zero checkers for IN_PARALLELISM arrays of IN_SIZE elements
*/


module nzc_group #(
    parameter IN_WIDTH = 32,
    parameter IN_SIZE = 4,
    parameter IN_PARALLELISM = 2
) (
    input [IN_WIDTH-1 :0] data_in[IN_SIZE*IN_PARALLELISM - 1 : 0],
    output [IN_PARALLELISM-1 :0] zero_flags
);
    for (genvar i = 0; i < IN_PARALLELISM; i = i + 1) begin
        // slice one block from data_in
        logic [IN_WIDTH-1 :0] current_data_in [IN_SIZE-1 :0];
        assign current_data_in = data_in[(i+1)*IN_SIZE - 1: i*IN_SIZE];
        nzc #(
            .IN_WIDTH (IN_WIDTH),
            .IN_SIZE (IN_SIZE)
        ) nzc_inst (
            .data_in (current_data_in),
            .zero_flag (zero_flags[i])
        );
    end
endmodule