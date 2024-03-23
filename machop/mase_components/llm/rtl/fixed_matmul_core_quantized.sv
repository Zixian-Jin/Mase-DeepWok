`timescale 1ns / 1ps
module fixed_matmul_core_quantized #(
    // input data_in1 = data_in; data_in2 = weight.
    parameter IN1_WIDTH = 8,
    parameter IN1_FRAC_WIDTH = 4,
    parameter IN2_WIDTH = 8,
    parameter IN2_FRAC_WIDTH = 4,
    parameter BIAS_WIDTH = 6,
    parameter BIAS_FRAC_WIDTH = 3,
    //output 
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 4,
    // define as nm * mk
    // rows refers to n, columns refers to mz
    parameter IN1_PARALLELISM = 4,
    parameter IN_SIZE = 1,
    parameter IN2_PARALLELISM = 3,
    //defines the dataflow parameter, used for linear layer
    parameter IN_DEPTH = 3,

    parameter HAS_BIAS = 0,
    parameter BIAS_PARALLELISM = IN1_PARALLELISM,
    parameter BIAS_SIZE = IN2_PARALLELISM,

    parameter OUT_ROWS = IN1_PARALLELISM, //rows
    parameter OUT_COLUMNS = IN2_PARALLELISM,  //cols

    parameter QUANTIZATION_WIDTH = 8
) (
    input clk,
    input rst,
    //input data
    input [IN1_WIDTH-1:0] data_in1[IN1_PARALLELISM * IN_SIZE - 1:0],
    input data_in1_valid,
    output data_in1_ready,
    //input weight
    input [IN2_WIDTH-1:0] data_in2[IN_SIZE * IN2_PARALLELISM - 1:0],
    input data_in2_valid,
    output data_in2_ready,
    //input bias
    input [BIAS_WIDTH-1:0] bias[BIAS_PARALLELISM * BIAS_SIZE - 1:0],
    input bias_valid,
    output bias_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[OUT_ROWS * OUT_COLUMNS - 1:0],
    output data_out_valid,
    input data_out_ready
);

/********* data_in_1 (data_in) quantization *******/
logic data_in1_int8_out_valid;
logic data_in1_int8_out_ready;
logic [QUANTIZATION_WIDTH-1:0] data_in1_int8 [IN1_PARALLELISM*IN_SIZE-1:0];
logic [IN1_WIDTH-1:0] data_in1_max_num;

quantizer_top #(
    .IN_WIDTH(IN1_WIDTH),  // FP16
    .IN_SIZE(IN_SIZE),  // in cols//TODO???? or IN_SIZE(IN1_PARALLELISM)????????????
    .IN_PARALLELISM(IN1_PARALLELISM), // in rows
    .OUT_WIDTH(QUANTIZATION_WIDTH) // int8
) quantizer_top_data_in_inst(
    .clk(clk),
    .rst(rst),
    // data_in
    .data_in(data_in1),
    .data_in_valid(data_in1_valid),
    .data_in_ready(data_in1_ready),
    // data_out
    .data_out(data_in1_int8),
    .data_out_valid(data_in1_int8_out_valid),
    .data_out_ready(data_in1_int8_out_ready),   
    .max_num(data_in1_max_num)  //TODO: change datatype // TODO: if max_num is 0, just skip this module?
);



/********* data_in_2 (weight) quantization *******/
logic data_in2_int8_out_valid;
logic data_in2_int8_out_ready;
logic [QUANTIZATION_WIDTH-1:0] data_in2_int8 [IN2_PARALLELISM*IN_SIZE-1:0];
logic [IN2_WIDTH-1:0] data_in2_max_num;

quantizer_top #(
    .IN_WIDTH(IN2_WIDTH),  // FP16
    .IN_SIZE(IN_SIZE),  // in cols//TODO???? or IN_SIZE(IN1_PARALLELISM)????????????
    .IN_PARALLELISM(IN2_PARALLELISM), // in rows

    .OUT_WIDTH(QUANTIZATION_WIDTH) // int8
) quantizer_top_weight_inst(
    .clk(clk),
    .rst(rst),
    // data_in
    .data_in(data_in2),
    .data_in_valid(data_in2_valid),
    .data_in_ready(data_in2_ready),
    // data_out
    .data_out(data_in2_int8),
    .data_out_valid(data_in2_int8_out_valid),
    .data_out_ready(data_in2_int8_out_ready),   
    .max_num(data_in2_max_num)  //TODO: change datatype // TODO: if max_num is 0, just skip this module?
);




/******* int8 multiplication *******/
// TODO: dummy bias
logic [QUANTIZATION_WIDTH-1: 0] bias_int8 [OUT_ROWS*OUT_COLUMNS-1 :0];
logic bias_int8_out_ready;
logic bias_int8_out_valid;
assign bias_int8_out_valid = data_in1_int8_out_valid;
assign bias_ready = data_in1_ready;  //TODO

localparam FMM_OUT_WIDTH = 17;  //TODO
logic [FMM_OUT_WIDTH-1: 0] fmm_out [OUT_ROWS*OUT_COLUMNS-1 :0];
logic fmm_out_ready, fmm_out_valid;

fixed_matmul_core #(
    .IN1_WIDTH (QUANTIZATION_WIDTH),
    .IN1_FRAC_WIDTH (0),
    .IN2_WIDTH (QUANTIZATION_WIDTH),
    .IN2_FRAC_WIDTH (0),
    .BIAS_WIDTH (QUANTIZATION_WIDTH),
    .BIAS_FRAC_WIDTH (0),
    .OUT_WIDTH (FMM_OUT_WIDTH),
    .OUT_FRAC_WIDTH (0),
    .IN1_PARALLELISM (IN1_PARALLELISM),
    .IN_SIZE (IN_SIZE),
    .IN2_PARALLELISM (IN2_PARALLELISM),
    .IN_DEPTH (IN_DEPTH),
    .HAS_BIAS (HAS_BIAS)
) fmm_int8 (
    .clk (clk),
    .rst (rst),
    .data_in1 (data_in1_int8),
    .data_in1_valid (data_in1_int8_out_valid),
    .data_in1_ready (data_in1_int8_out_ready),

    .data_in2 (data_in2_int8),
    .data_in2_valid (data_in2_int8_out_valid),
    .data_in2_ready (data_in2_int8_out_ready),

    .bias (bias_int8),
    .bias_valid (bias_int8_out_valid),
    .bias_ready (bias_int8_out_ready),

    .data_out (fmm_out),
    .data_out_valid (fmm_out_valid),
    .data_out_ready (fmm_out_ready)
);




/******* max num multiplication *******/
localparam MAX_NUM_WIDTH = IN1_WIDTH + IN2_WIDTH;  //TODO
logic [MAX_NUM_WIDTH-1: 0] max_num; 
// logic max_num_ready, max_num_valid;
// join2 #(
// ) max_num_join (
//     .data_in_valid ({data_in1_int8_out_valid, data_in2_int8_out_valid}),
//     .data_in_ready ({data_in1_int8_out_ready, data_in2_int8_out_ready}),
//     .data_out_valid (max_num_valid),
//     .data_out_ready (max_num_ready)
// );
fixed_mult #(
    .IN_A_WIDTH (IN1_WIDTH),
    .IN_B_WIDTH (IN2_WIDTH)
) max_num_mult_inst (
    .data_a (data_in1_max_num),
    .data_b (data_in2_max_num),
    .product (max_num)
);

logic [MAX_NUM_WIDTH-1: 0] max_num_buffered;
logic max_num_buffered_ready, max_num_buffered_valid;
localparam FMM_DELAY = IN_SIZE * 10;  //TODO: fifo depth too large?
fifo #(
    .DEPTH (FMM_DELAY+1),
    .DATA_WIDTH (MAX_NUM_WIDTH)
) data_in_fifo_inst (
    .clk (clk),
    .rst (rst),
    .in_data (max_num),
    .in_valid (data_in1_int8_out_valid),   //TODO: assume din1_max_num & din2_max_num arrive in sync
    // .in_ready (data_in1_int8_out_ready),
    .out_data (max_num_buffered),
    .out_valid (max_num_buffered_valid),
    .out_ready (max_num_buffered_ready)
    // .empty (fifo_empty)
);

/******* Dequantizer *******/
logic dequantizer_in_valid, dequantizer_in_ready;
join2 #(
) dequant_join (
    .data_in_valid ({max_num_buffered_valid, fmm_out_valid}),
    .data_in_ready ({max_num_buffered_ready, fmm_out_ready}),
    .data_out_valid (dequantizer_in_valid),
    .data_out_ready (dequantizer_in_ready)
);

dequantizer #(
    .IN_WIDTH (FMM_OUT_WIDTH),
    .IN_SIZE (OUT_COLUMNS),
    .IN_PARALLELISM (OUT_ROWS),
    .OUT_WIDTH (OUT_WIDTH),
    .MAX_NUM_WIDTH (MAX_NUM_WIDTH),
    .QUANTIZATION_WIDTH (QUANTIZATION_WIDTH*2)
) dequant_inst(
    .clk (clk),
    .rst (rst),
    .data_in (fmm_out),
    .data_in_valid (dequantizer_in_valid),
    .data_in_ready (dequantizer_in_ready),
    .max_num (max_num),
    .data_out (data_out),
    .data_out_valid (data_out_valid),
    .data_out_ready (data_out_ready)
);


endmodule