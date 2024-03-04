`timescale 1ns / 1ps

/*
 * Constrained by WEIGHT_PARALLELISM_DIM_0 == DATA_OUT_PARALLELISM_DIM_0
 *
*/

module gather #(
    /* verilator lint_off UNUSEDPARAM */
    parameter THRES = 30,

    // parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,
    parameter IN_PARALLELISM = 1,
    parameter OUT_COLUMNS = IN_SIZE
) (
    input clk,
    input rst,

    // input port for weight
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE * IN_PARALLELISM-1:0],
    input data_in_valid,
    output data_in_ready,

    output [0:0] data_out [IN_SIZE-1:0],
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

    // Assumes that if a large number occurs at the first row, then the rest entries in that column are all large numbers
    for (genvar j = 0; j < IN_SIZE; j = j + 1) begin: COL
        fp16_comparator #(
            .THRES(THRES)
        ) fp16_comp_inst(
            .data_in(data_in[j]),
            .result(data_out[j])
        );
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
    assign data_out_valid = !rst;
endmodule
