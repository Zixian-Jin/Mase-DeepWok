/*
Module      : compressed_row_fetcher
Description : This module is dedicated for sparse hardware. 
              It fetches a specific dense row from a dense matrix. 
              out_data is padded with zero if the actual dense row size is smaller than NZN_ROW
              Both data are stored in COO format.
*/
`timescale 1ns/1ps
module csr_row_fetcher #(
    parameter IN_SIZE = 4,  // total num of elements in in_data
    parameter FETCH_SIZE = 2,   // num of elements to be selected from in_data,
    parameter IN_PARALLELISM = 3,  // num of rows
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16
) (
    // input matrix in COO format
    input logic [DATA_WIDTH-1:0] in_data [0 : IN_SIZE-1],  
    input logic [ADDR_WIDTH-1:0] in_index [0 : IN_SIZE-1],
    input logic [ADDR_WIDTH-1:0] in_bounds [0 : (IN_PARALLELISM+1)-1],

    input logic [ADDR_WIDTH-1:0] fetch_row,  // which row to fetch from in_data

    // output row in COO format
    output logic [DATA_WIDTH-1:0] out_data [0 : FETCH_SIZE-1],
    output logic [ADDR_WIDTH-1:0] out_index [0 : FETCH_SIZE-1]
);
    integer i, j;
    logic [ADDR_WIDTH-1:0] ind_offset;
    logic [ADDR_WIDTH-1:0] ind_upper_bound;

    assign ind_offset = in_bounds[fetch_row];
    assign ind_upper_bound = in_bounds[fetch_row + 1];

    for (genvar i = 0; i < FETCH_SIZE; i += 1) begin
        logic [ADDR_WIDTH-1:0] addr;
        logic valid;
        assign addr = ind_offset + i;
        assign valid = (addr < ind_upper_bound) ? 1 : 0;

        assign out_data[i] = valid ? in_data[addr] : 0;
        assign out_index[i] = valid ? in_index[addr] : -1;
    end

    //     for (i = ind_start; i < ind_end; i += 1) begin
    //             out_data[j] = in_data[i];
    //             out_index[j] = in_index[i];
    //             j = j + 1;
    //     end
    // end


endmodule