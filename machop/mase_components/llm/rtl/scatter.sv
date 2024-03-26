`timescale 1ns / 1ps
/*
 * ASSUMPTION: OUT_SMALL_COLUMNS == # zeros in ind_table
 */
module scatter #(
    /* verilator lint_off UNUSEDPARAM */

    // parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 1  // in rows
    // parameter OUT_LARGE_COLUMNS = 2,
    // parameter OUT_SMALL_COLUMNS = IN_SIZE - OUT_LARGE_COLUMNS
) (
    input clk,
    input rst,

    // input port for weight
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE * IN_PARALLELISM-1:0],
    input data_in_valid,
    output data_in_ready,

    output [IN_WIDTH-1:0] data_out_large [IN_SIZE * IN_PARALLELISM-1:0],
    output [IN_WIDTH-1:0] data_out_small [IN_SIZE * IN_PARALLELISM-1:0],
    output data_out_valid,
    input data_out_ready
);
    // Check all entries of the input matrix
    for (genvar i = 0; i < IN_PARALLELISM; i = i + 1) begin: ROW 
        // Parse flattened data_in
        logic [IN_WIDTH-1 :0] current_data_in [IN_SIZE-1 :0];
        assign current_data_in = data_in[IN_SIZE*(i+1)-1 :IN_SIZE*i];
        
        // Parse flattened data_out
        logic [IN_WIDTH-1 :0] current_data_out_large [IN_SIZE-1 :0];
        logic [IN_WIDTH-1 :0] current_data_out_small [IN_SIZE-1 :0];
        assign data_out_large[IN_SIZE*(i+1)-1 :IN_SIZE*i] = current_data_out_large[IN_SIZE-1 :0];
        assign data_out_small[IN_SIZE*(i+1)-1 :IN_SIZE*i] = current_data_out_small[IN_SIZE-1 :0];

        for (genvar j = 0; j < IN_SIZE; j = j + 1) begin: COL

            logic is_outlier;
            fp16_comparator #(
                .IN_WIDTH (IN_WIDTH)
            ) fp16_cmp_inst (
                .data_in (current_data_in[j]),
                .result (is_outlier)
            );  

            always_comb begin
                if (is_outlier == 1'b0) begin
                    // small number, allocated to small column
                    current_data_out_small[j] = current_data_in[j];
                    current_data_out_large[j] = 0;
                end else begin
                    // large number
                    current_data_out_small[j] = 0;
                    current_data_out_large[j] = current_data_in[j];                
                end
            end
        end
        
        /* An alternativa but aborted stratey: process entries of one row in serial */
        // always_ff @(posedge clk) begin: entryAssignment
        //     if (ind_table[counter] == 1'b0) begin
        //         // small number, allocated to a small column
        //         current_data_out_small[counter] <= current_entry;
        //         current_data_out_large[counter] <= 0;
        //     end else begin
        //         // large number
        //         current_data_out_small[counter] <= 0;
        //         current_data_out_large[counter] <= current_entry;
        //     end
        // end


    end

    assign data_out_valid = !rst && data_in_valid; //&& (ROW[0].counter == IN_SIZE);
    assign data_in_ready = !rst && data_out_ready;
endmodule
