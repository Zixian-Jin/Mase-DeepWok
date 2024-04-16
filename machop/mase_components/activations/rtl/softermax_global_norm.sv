/*
Module      : softermax_global_norm
Description : This module implements the second section of the softermax compute
              pipeline which calculates renormalizes all local values and
              calculates the final.

              Refer to bottom half of Fig. 4.a) and 4.b) in the Softermax Paper.
              https://arxiv.org/abs/2103.09301
*/

`timescale 1ns/1ps

module softermax_global_norm #(
    // Input shape dimensions
    parameter TOTAL_DIM = 16,
    parameter PARALLELISM = 4,

    // Widths
    parameter IN_VALUE_WIDTH = 8,
    parameter IN_VALUE_FRAC_WIDTH = 4,
    parameter IN_MAX_WIDTH = 8,
    parameter IN_MAX_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 7
) (
    input  logic clk,
    input  logic rst,

    // Input streaming interface
    input  logic [IN_VALUE_WIDTH-1:0]  in_values [PARALLELISM-1:0],
    input  logic [IN_MAX_WIDTH-1:0]    in_max,
    input  logic                       in_valid,
    output logic                       in_ready,

    // Output streaming interface with pow of 2 values & max
    output logic [OUT_WIDTH-1:0]       out_data,
    output logic                       out_valid,
    input  logic                       out_ready
);

// -----
// Parameters
// -----

localparam DEPTH = TOTAL_DIM / PARALLELISM;

localparam SUBTRACT_WIDTH = IN_MAX_WIDTH + 1;
localparam SUBTRACT_FRAC_WIDTH = IN_MAX_FRAC_WIDTH;

localparam ADDER_TREE_WIDTH = $clog2(PARALLELISM) + IN_VALUE_WIDTH;
localparam ADDER_TREE_FRAC_WIDTH = IN_VALUE_FRAC_WIDTH;

localparam ACC_WIDTH = $clog2(DEPTH) + ADDER_TREE_WIDTH;
localparam ACC_FRAC_WIDTH = ADDER_TREE_FRAC_WIDTH;


// -----
// Wires
// -----

logic [IN_MAX_WIDTH-1:0] local_max_out;
logic local_max_in_valid, local_max_in_ready;
logic local_max_out_valid, local_max_out_ready;

logic [IN_MAX_WIDTH-1:0] global_max_out;
logic global_max_in_valid, global_max_in_ready;
logic global_max_out_valid, global_max_out_ready;

logic [IN_VALUE_WIDTH-1:0] local_values_out [PARALLELISM-1:0];
logic local_values_in_valid, local_values_in_ready;
logic local_values_out_valid, local_values_out_ready;

logic [SUBTRACT_WIDTH-1:0] subtract_in_data [PARALLELISM-1:0];
logic [SUBTRACT_WIDTH-1:0] subtract_out_data [PARALLELISM-1:0];
logic subtract_in_valid;
logic subtract_in_ready [PARALLELISM-1:0];
logic subtract_out_valid [PARALLELISM-1:0];
logic subtract_out_ready;

logic [IN_VALUE_WIDTH-1:0] shift_in_data [PARALLELISM-1:0];
logic [IN_VALUE_WIDTH-1:0] shift_out_data [PARALLELISM-1:0];
logic shift_in_valid;
logic shift_in_ready [PARALLELISM-1:0];
logic shift_out_valid [PARALLELISM-1:0];
logic shift_out_ready;

logic [IN_VALUE_WIDTH-1:0] adjusted_values_in_data [PARALLELISM-1:0];
logic [IN_VALUE_WIDTH-1:0] adjusted_values_out_data [PARALLELISM-1:0];
logic adjusted_values_in_valid, adjusted_values_in_ready;
logic adjusted_values_out_valid, adjusted_values_out_ready;

logic [IN_VALUE_WIDTH-1:0] adder_tree_in_data [PARALLELISM-1:0];
logic [ADDER_TREE_WIDTH-1:0] adder_tree_out_data;
logic adder_tree_in_valid, adder_tree_in_ready;
logic adder_tree_out_valid, adder_tree_out_ready;

logic [ACC_WIDTH-1:0] acc_out_data;
logic acc_out_valid, acc_out_ready;

// -----
// Modules
// -----

splitn #(
    .N(3)
) input_split (
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out_valid({local_max_in_valid,
                     global_max_in_valid,
                     local_values_in_valid}),
    .data_out_ready({local_max_in_ready,
                     global_max_in_ready,
                     local_values_in_ready})
);

fifo #(
    .DATA_WIDTH(IN_MAX_WIDTH),
    .SIZE(DEPTH + 4) // TODO: resize +1 ?
) local_max_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(in_max),
    .in_valid(local_max_in_valid),
    .in_ready(local_max_in_ready),
    .out_data(local_max_out),
    .out_valid(local_max_out_valid),
    .out_ready(local_max_out_ready)
);

comparator_accumulator #(
    .DATA_WIDTH(IN_MAX_WIDTH),
    .DEPTH(DEPTH),
    .MAX1_MIN0(1),
    .SIGNED(1)
) global_max_accumulator (
    .clk(clk),
    .rst(rst),
    .in_data(in_max),
    .in_valid(global_max_in_valid),
    .in_ready(global_max_in_ready),
    .out_data(global_max_out),
    .out_valid(global_max_out_valid),
    .out_ready(global_max_out_ready)
);

matrix_fifo #(
    .DATA_WIDTH(IN_VALUE_WIDTH),
    .DIM0(PARALLELISM),
    .DIM1(1),
    .FIFO_SIZE(DEPTH + 10) // TODO: resize?
) local_values_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(in_values),
    .in_valid(local_values_in_valid),
    .in_ready(local_values_in_ready),
    .out_data(local_values_out),
    .out_valid(local_values_out_valid),
    .out_ready(local_values_out_ready)
);


join2 subtract_join (
    .data_in_valid({global_max_out_valid, local_max_out_valid}),
    .data_in_ready({global_max_out_ready, local_max_out_ready}),
    .data_out_valid(subtract_in_valid),
    .data_out_ready(subtract_in_ready[0])
);

// Batched subtract
for (genvar i = 0; i < PARALLELISM; i++) begin : subtract
    assign subtract_in_data[i] = $signed(global_max_out) - $signed(local_max_out[i]);

    skid_buffer #(
        .DATA_WIDTH(SUBTRACT_WIDTH)
    ) sub_reg (
        .clk(clk),
        .rst(rst),
        .data_in(subtract_in_data[i]),
        .data_in_valid(subtract_in_valid),
        .data_in_ready(subtract_in_ready[i]),
        .data_out(subtract_out_data[i]),
        .data_out_valid(subtract_out_valid[i]),
        .data_out_ready(subtract_out_ready)
    );
end

join2 shift_join (
    .data_in_valid({local_values_out_valid, subtract_out_valid[0]}),
    .data_in_ready({local_values_out_ready, subtract_out_ready}),
    .data_out_valid(shift_in_valid),
    .data_out_ready(shift_in_ready[0])
);


// Batched shift
for (genvar i = 0; i < PARALLELISM; i++) begin : shift
    assign shift_in_data[i] = local_values_out[i] >> subtract_out_data[i];

    skid_buffer #(
        .DATA_WIDTH(SUBTRACT_WIDTH)
    ) sub_reg (
        .clk(clk),
        .rst(rst),
        .data_in(shift_in_data[i]),
        .data_in_valid(shift_in_valid),
        .data_in_ready(shift_in_ready[i]),
        .data_out(shift_out_data[i]),
        .data_out_valid(shift_out_valid[i]),
        .data_out_ready(shift_out_ready)
    );
end

split2 norm_split (
    .data_in_valid(shift_out_valid[0]),
    .data_in_ready(shift_out_ready),
    .data_out_valid({adjusted_values_in_valid, adder_tree_in_valid}),
    .data_out_ready({adjusted_values_in_ready, adder_tree_in_ready})
);

assign adjusted_values_in_data = shift_out_data;
assign adder_tree_in_data = shift_out_data;

matrix_fifo #(
    .DATA_WIDTH(IN_VALUE_WIDTH),
    .DIM0(PARALLELISM),
    .DIM1(1),
    .FIFO_SIZE(DEPTH + 32) // TODO: resize?
) adjusted_values_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(adjusted_values_in_data),
    .in_valid(adjusted_values_in_valid),
    .in_ready(adjusted_values_in_ready),
    .out_data(adjusted_values_out_data),
    .out_valid(adjusted_values_out_valid),
    .out_ready(adjusted_values_out_ready)
);

fixed_adder_tree #(
    .IN_SIZE(PARALLELISM),
    .IN_WIDTH(IN_VALUE_WIDTH)
) adder_tree (
    .clk(clk),
    .rst(rst),
    .data_in(adder_tree_in_data),
    .data_in_valid(adder_tree_in_valid),
    .data_in_ready(adder_tree_in_ready),
    .data_out(adder_tree_out_data),
    .data_out_valid(adder_tree_out_valid),
    .data_out_ready(adder_tree_out_ready)
);

fixed_accumulator #(
    .IN_DEPTH(DEPTH),
    .IN_WIDTH(ADDER_TREE_WIDTH)
) norm_accumulator (
    .clk(clk),
    .rst(rst),
    .data_in(adder_tree_out_data),
    .data_in_valid(adder_tree_out_valid),
    .data_in_ready(adder_tree_out_ready),
    .data_out(acc_out_data),
    .data_out_valid(acc_out_valid),
    .data_out_ready(acc_out_ready)
);


// lpw reciprocal


endmodule
