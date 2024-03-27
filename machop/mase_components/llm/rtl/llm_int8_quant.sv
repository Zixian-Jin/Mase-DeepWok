`timescale 1ns / 1ps
module llm_int8_quant #(
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 5, // in rows

    parameter WEIGHT_WIDTH = IN_WIDTH,  // FP16
    parameter WEIGHT_SIZE = IN_SIZE,  // in rows
    parameter WEIGHT_PARALLELISM = 1, // in cols

    parameter HAS_BIAS = 0,
    parameter BIAS_WIDTH = IN_WIDTH,  // FP16
    parameter BIAS_PARALLELISM = IN_PARALLELISM,  // aborted
    parameter BIAS_SIZE = WEIGHT_PARALLELISM, // aborted

    parameter OUT_WIDTH = 2*IN_WIDTH, // TODO
    parameter OUT_ROWS = IN_PARALLELISM,
    parameter OUT_COLUMNS = WEIGHT_PARALLELISM,

    parameter IN_DEPTH = 3,
    parameter QUANTIZATION_WIDTH = 8,  // int8
    parameter MAX_LARGE_NUMBERS = 10,  // for scattering: max number of outliers in the HP matrix
    parameter LARGE_NUMBER_THRES = 127  // for scattering: numbers larger than (BUT NOT EQUAL TO) this thres are counted as outliers
) (
    input clk,
    input rst,

    // data_in
    input  [IN_WIDTH-1:0] data_in      [IN_PARALLELISM * IN_SIZE-1: 0],
    input data_in_valid,
    output data_in_ready,
    
    // weight
    input  [IN_WIDTH-1:0] weight      [WEIGHT_SIZE * WEIGHT_PARALLELISM-1: 0],
    input weight_valid,
    output weight_ready,

    // output
    output [OUT_WIDTH-1:0] data_out      [OUT_ROWS * OUT_COLUMNS - 1 :0],

    output data_out_valid,
    input data_out_ready    
);

    logic data_in_out_valid;
    logic data_in_out_ready;
    logic  [IN_WIDTH-1:0] data_in_large      [IN_SIZE * IN_PARALLELISM-1: 0];
    logic  [IN_WIDTH-1:0] data_in_small      [IN_SIZE * IN_PARALLELISM-1: 0];
    scatter #(
        .IN_WIDTH (IN_WIDTH),
        .IN_SIZE (IN_SIZE),
        .IN_PARALLELISM (IN_PARALLELISM),
        .MAX_LARGE_NUMBERS (MAX_LARGE_NUMBERS),
        .LARGE_NUMBER_THRES (LARGE_NUMBER_THRES)
    ) scatter_data_in(
        .clk(clk),
        .rst(rst),
        // input port for weight
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .data_in_ready(data_in_ready),  // TODO
        
        .data_out_large(data_in_large),
        .data_out_small(data_in_small),
        .data_out_valid(data_in_out_valid),
        .data_out_ready(data_in_out_ready)
    );


    logic weight_out_valid;
    logic weight_out_ready;
    assign weight_out_valid = weight_valid;
    assign weight_ready = weight_out_ready;

    // logic  [IN_WIDTH-1:0] weight_large      [WEIGHT_PARALLELISM * WEIGHT_SIZE-1: 0];
    // logic  [IN_WIDTH-1:0] weight_small      [WEIGHT_PARALLELISM * WEIGHT_SIZE-1: 0];
    // scatter #(
    //     .IN_WIDTH (WEIGHT_WIDTH),
    //     .IN_SIZE (WEIGHT_PARALLELISM),
    //     .IN_PARALLELISM (WEIGHT_SIZE)
    // ) scatter_weight(
    //     .clk(clk),
    //     .rst(rst),
    //     // input port for weight
    //     .data_in(weight),
    //     .data_in_valid(weight_in_valid),
    //     .data_in_ready(weight_in_ready),
        
    //     .data_out_large(weight_large),
    //     .data_out_small(weight_small),
    //     .data_out_valid(weight_out_valid),
    //     .data_out_ready(weight_out_ready)
    // );


    // set dummy bias signals 
    logic [BIAS_WIDTH-1 :0] bias[BIAS_PARALLELISM * BIAS_SIZE - 1 : 0];
    logic bias_valid = 1'b1;
    logic bias_ready = 1'b1;


    logic [OUT_WIDTH-1:0] matmul_large [OUT_ROWS * OUT_COLUMNS - 1: 0];
    logic [OUT_WIDTH-1:0] matmul_small [OUT_ROWS * OUT_COLUMNS - 1: 0];

    // input control signals for fmm modules
    logic matmul_large_in_valid, matmul_large_in_ready;
    logic matmul_small_in_valid, matmul_small_in_ready;

    // output control signals for fmm modules
    logic matmul_large_out_valid, matmul_large_out_ready;
    logic matmul_small_out_valid, matmul_small_out_ready;


    split2 #(
    ) matmul_large_small_in_split(
        .data_in_valid (data_in_out_valid),
        .data_in_ready (data_in_out_ready),
        .data_out_valid ({matmul_large_in_valid, matmul_small_in_valid}),
        .data_out_ready ({matmul_large_in_ready, matmul_small_in_ready})
    );

    /* LARGE matrix */
    fixed_matmul_core #(
        .IN1_WIDTH(IN_WIDTH),
        .IN1_FRAC_WIDTH(0),
        .IN2_WIDTH(WEIGHT_WIDTH),
        .IN2_FRAC_WIDTH(0),
        .BIAS_WIDTH(BIAS_WIDTH),
        .BIAS_FRAC_WIDTH(0),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(0),
        .IN1_PARALLELISM(IN_PARALLELISM),
        .IN_SIZE(IN_SIZE),
        .IN2_PARALLELISM(WEIGHT_PARALLELISM),
        .IN_DEPTH(IN_DEPTH),
        .HAS_BIAS(HAS_BIAS)
    ) inst_fmmc_large (
        .clk(clk),
        .rst(rst),
        .data_in1(data_in_large),
        .data_in1_valid(matmul_large_in_valid),
        .data_in1_ready(matmul_large_in_ready),
        .data_in2(weight),
        .data_in2_valid(weight_out_valid),
        .data_in2_ready(weight_out_ready),
        .bias(bias),
        .bias_valid(bias_valid),
        .bias_ready(bias_ready),
        .data_out(matmul_large),
        .data_out_valid(matmul_large_out_valid),
        .data_out_ready(matmul_large_out_ready)
    );

    /* SMALL matrix */
    fixed_matmul_core_quantized #(
        .IN1_WIDTH(IN_WIDTH),
        .IN1_FRAC_WIDTH(0),
        .IN2_WIDTH(WEIGHT_WIDTH),
        .IN2_FRAC_WIDTH(0),
        .BIAS_WIDTH(BIAS_WIDTH),
        .BIAS_FRAC_WIDTH(0),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(0),
        .IN1_PARALLELISM(IN_PARALLELISM),
        .IN_SIZE(IN_SIZE),
        .IN2_PARALLELISM(WEIGHT_PARALLELISM),
        .IN_DEPTH(IN_DEPTH),
        .HAS_BIAS(HAS_BIAS),
        .QUANTIZATION_WIDTH(QUANTIZATION_WIDTH)
    ) inst_fmmc_small (
        .clk(clk),
        .rst(rst),
        .data_in1(data_in_small),
        .data_in1_valid(matmul_small_in_valid),
        .data_in1_ready(matmul_small_in_ready),
        .data_in2(weight),
        .data_in2_valid(weight_out_valid),
        .data_in2_ready(weight_out_ready),
        .bias(bias),
        .bias_valid(bias_valid),
        .bias_ready(bias_ready),
        .data_out(matmul_small),
        .data_out_valid(matmul_small_out_valid),
        .data_out_ready(matmul_small_out_ready)
    );

    /* HP and LP Matrix Gather */
    for (genvar i = 0; i < OUT_ROWS * OUT_COLUMNS; i = i + 1) begin: GATHER
        assign data_out[i] = matmul_large[i] + matmul_small[i];
    end


    join2 #(
    ) matmul_large_small_out_join (
        .data_in_valid ({matmul_large_out_valid, matmul_small_out_valid}),
        .data_in_ready ({matmul_large_out_ready, matmul_small_out_ready}),
        .data_out_valid (data_out_valid),
        .data_out_ready (data_out_ready)
    );
    // assign data_in_ready = !rst;
    // assign data_out_valid = !rst && matmul_large_valid && matmul_small_valid;  // TODO: join
    // assign matmul_large_ready = data_out_ready;
    // assign matmul_small_ready = data_out_ready;

endmodule