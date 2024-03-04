`timescale 1ns / 1ps
/*
 * ASSUMPTION: OUT_SMALL_COLUMNS == # zeros in ind_table
 */
module scatter #(
    /* verilator lint_off UNUSEDPARAM */

    // parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 1,  // in rows
    parameter OUT_LARGE_COLUMNS = 2,
    parameter OUT_SMALL_COLUMNS = IN_SIZE - OUT_LARGE_COLUMNS
) (
    input clk,
    input rst,

    // input port for weight
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE * IN_PARALLELISM-1:0],
    input data_in_valid,
    output data_in_ready,

    input [0:0] ind_table [IN_SIZE-1:0],

    output [IN_WIDTH-1:0] data_out_large [OUT_LARGE_COLUMNS * IN_PARALLELISM-1:0],
    output [IN_WIDTH-1:0] data_out_small [OUT_SMALL_COLUMNS * IN_PARALLELISM-1:0],
    output data_out_valid,
    input data_out_ready
);

//   logic fdp_join_valid, fdp_join_ready;
//   join2 #() fdp_join_inst (
//       .data_in_ready ({weight_ready, data_in_0_ready}),
//       .data_in_valid ({weight_valid, data_in_0_valid}),
//       .data_out_valid(fdp_join_valid),
//       .data_out_ready(fdp_join_ready)
//   );
    logic [IN_WIDTH-1:0] reg_out_large [OUT_LARGE_COLUMNS * IN_PARALLELISM-1:0];
    logic [IN_WIDTH-1:0] reg_out_small [OUT_SMALL_COLUMNS * IN_PARALLELISM-1:0];

    // Assumes that if a large number occurs at the first row, then the rest entries in that column are all large numbers
    for (genvar i = 0; i < IN_PARALLELISM; i = i + 1) begin: ROW 
        logic [IN_WIDTH-1 :0] current_data_in [IN_SIZE-1 :0];
        assign current_data_in = data_in[IN_SIZE*(i+1)-1: IN_SIZE*i];

        logic [IN_SIZE-1 :0] counter;  // counter max value == IN_SIZE
        initial counter = 0;
        logic [IN_WIDTH-1 :0] current_entry;

        logic [OUT_LARGE_COLUMNS-1 :0] counter_large;  // counter max value == OUT_LARGE_COLUMNS
        logic [OUT_SMALL_COLUMNS-1 :0] counter_small;  // counter max value == OUT_SMALL_COLUMNS

        always_ff @(posedge clk) begin: entryAssignment
            if (counter == IN_SIZE) begin
            end else begin
                current_entry <= current_data_in[counter];
                if (ind_table[counter] == 1'b0) begin
                    // small number, allocated to a small column
                    reg_out_small[counter_small] <= current_entry;
                end else begin
                    // large number
                    reg_out_large[counter_large] <= current_entry;
                end
            end
        end

        always_ff @(posedge clk) begin: CounterUpdate
            if (counter == IN_SIZE) begin
                counter <= 0;
            end else begin
                counter <= counter + 1;

                if (counter_small == OUT_SMALL_COLUMNS) counter_small <= 0;
                else counter_small <= counter_small + 1;

                if (counter_large == OUT_LARGE_COLUMNS) counter_large <= 0;
                else counter_large <= counter_large + 1;
            end
        end
    end

    // always_ff @(posedge clk) begin: RST
    //     if (rst) begin
    //         data_in_ready <= 0;
    //         data_out_valid <= 0;
    //     end else begin
    //         data_in_ready <= 1;
    //         data_out_valid <= 1;
    //     end
    // end
    assign data_in_ready = !rst;
    assign data_out_valid = !rst && (ROW[0].counter == IN_SIZE);
    assign data_out_small = reg_out_small;
    assign data_out_large = reg_out_large;
endmodule
