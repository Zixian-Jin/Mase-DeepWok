/*
Module      : compressed_row_fetcher
Description : This module is dedicated for sparse hardware. 
              It fetches a specific dense row from a dense matrix. 
              out_data is padded with zero if the actual dense row size is smaller than NZN_ROW
              Both data are stored in COO format.
*/
`timescale 1ns/1ps
module coo_row_fetcher #(
    parameter IN_SIZE = 4,  // total num of elements in in_data
    parameter FETCH_SIZE = 2,   // num of elements to be selected from in_data
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16
) (
    // input matrix in COO format
    input logic [DATA_WIDTH-1:0] in_data [IN_SIZE-1:0],  
    input logic [ADDR_WIDTH-1:0] in_row_table [IN_SIZE-1:0],
    input logic [ADDR_WIDTH-1:0] in_col_table [IN_SIZE-1:0],

    input logic [ADDR_WIDTH-1:0] fetch_row_index,  // which row to fetch from in_data

    // output row in COO format
    output logic [DATA_WIDTH-1:0] out_data [FETCH_SIZE-1:0],
    output logic [ADDR_WIDTH-1:0] out_row_table [FETCH_SIZE-1:0],
    output logic [ADDR_WIDTH-1:0] out_col_table [FETCH_SIZE-1:0]
);
    integer i, j;

    always_comb begin
        // initialization
        i = 0;
        j = 0;
        for (int k = 0; k < FETCH_SIZE; k += 1) begin
            out_data[k] = 0;
            out_row_table[k] = 0;
            out_col_table[k] = 0;
        end

        for (i = 0; i < IN_SIZE; i += 1) begin
            if (in_row_table[i] == fetch_row_index) begin
                out_data[j] = in_data[i];
                out_row_table[j] = in_row_table[i];
                out_col_table[j] = in_col_table[i];
                j = j + 1;
            end else begin
                j = j + 0;
            end            
        end
    end


endmodule