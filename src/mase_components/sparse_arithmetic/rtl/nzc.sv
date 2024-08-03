/* 
 * Module: nzc
 * Description: non-zero check for an array of IN_SIZE elements
*/

`timescale 1ns / 1ps

// Module: non-zero checker
module nzc #(
    parameter IN_WIDTH = 32,
    parameter IN_SIZE = 4
) (
    input [IN_WIDTH-1 :0] data_in[IN_SIZE-1 :0],
    output zero_flag
);
    // TODO
    assign zero_flag = (data_in[0] == 0) && (data_in[1] == 0);
endmodule