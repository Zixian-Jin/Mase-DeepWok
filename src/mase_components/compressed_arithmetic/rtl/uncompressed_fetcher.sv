/*
Module      : uncompressed_fetcher
Description : This module is dedicated for sparse hardware. 
              It fetches FETCH_SIZE elements from a data vector of length DATA_SIZE
              based on the address vector in_addr_table in COO format.
*/
`timescale 1ns/1ps
module uncompressed_fetcher #(
    parameter DATA_SIZE = 4,  // total num of elements in in_data
    parameter FETCH_SIZE = 2,   // num of elements to be selected from in_data
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16
) (
    input logic [ADDR_WIDTH-1:0] in_addr_table [0 : FETCH_SIZE-1],
    input logic [DATA_WIDTH-1:0] in_data [0 : DATA_SIZE-1],
    output logic [DATA_WIDTH-1:0] out_data [0 : FETCH_SIZE-1]
);
    for (genvar i = 0; i < FETCH_SIZE; i += 1) begin
        logic [ADDR_WIDTH-1:0] current_addr;
        assign current_addr = in_addr_table[i];
        assign out_data[i] = ($signed(current_addr) < 0) ? 0 : in_data[current_addr];
    end

endmodule