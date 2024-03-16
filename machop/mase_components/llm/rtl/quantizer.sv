`timescale 1ns / 1ps

module quantizer #(
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 1, // in rows

    parameter OUT_WIDTH = 8, // int8
    parameter OUT_ROWS = IN_PARALLELISM,
    parameter OUT_COLUMNS = IN_SIZE,

    parameter MAX_NUM_WIDTH = IN_WIDTH
) (
    input clk,
    input rst,

    // data_in
    input  [IN_WIDTH-1:0] data_in      [IN_PARALLELISM * IN_SIZE-1: 0],
    input data_in_valid,
    output data_in_ready,

    // data_out
    output [OUT_WIDTH-1:0] data_out      [OUT_ROWS * OUT_COLUMNS - 1 :0],

    output data_out_valid,
    input data_out_ready,   

    output [MAX_NUM_WIDTH-1:0] max_num  //TODO: change datatype
);

// logic [IN_WIDTH-1:0] max_num;// TODO: if max_num is 0, just skip this module?


localparam SCALE_FACTOR_FRAC_WIDITH = 16;
localparam SCALE_FACTOR_WIDTH = 32;
logic [SCALE_FACTOR_WIDTH-1:0] scale_factor;//TODO: change datatype
logic [MAX_NUM_WIDTH-1:0] reg_max_num;
logic [MAX_NUM_WIDTH-1:0] reg_max_num_abs;

assign reg_max_num_abs = ($signed(reg_max_num) > 0) ? reg_max_num : -reg_max_num;
assign max_num = reg_max_num_abs;


fixed_comparator_tree #(
    .IN_SIZE(IN_SIZE*IN_PARALLELISM),
    .IN_WIDTH(IN_WIDTH),
)fixed_comparator_tree_inst(
    .clk(clk),
    .rst(rst),
    /* verilator lint_on UNUSEDSIGNAL */
    .data_in(data_in),
    .data_in_valid(data_in_valid),
    .data_in_ready(data_in_ready),
    .data_out(reg_max_num),
    .data_out_valid(data_out_valid),
    .data_out_ready(data_out_ready)
);


// assign scale_factor = ((1 << (OUT_WIDTH-1))-1) << (SCALE_FACTOR_FRAC_WIDITH); //TODO?? reshape: make its width 2*IN_SIZE and do multiply first then divide
logic [SCALE_FACTOR_WIDTH-1:0] temp;
assign temp = (8'b01111111) << (SCALE_FACTOR_FRAC_WIDITH);  // temp = 127 << 8 = 0x7F00
assign scale_factor = temp/reg_max_num_abs; //TODO?? reshape: make its width 2*IN_SIZE and do multiply first then divide

// assign scale_factor = max_mum >> OUT_WIDTH; 

logic [IN_WIDTH + SCALE_FACTOR_WIDTH-1:0] data_out_unrounded      [OUT_ROWS * OUT_COLUMNS - 1 :0];

for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: QUANTIZE
    fixed_mult #(
    .IN_A_WIDTH(IN_WIDTH),
    .IN_B_WIDTH(SCALE_FACTOR_WIDTH)
    ) fixed_mult_inst(
    .data_a(data_in[i]),
    .data_b(scale_factor),
    .product(data_out_unrounded[i])
    );
end



fixed_rounding #(
    .IN_SIZE(OUT_ROWS*OUT_COLUMNS),
    .IN_WIDTH(IN_WIDTH + SCALE_FACTOR_WIDTH),
    .IN_FRAC_WIDTH(SCALE_FACTOR_FRAC_WIDITH),
    .OUT_WIDTH(OUT_WIDTH),
    .OUT_FRAC_WIDTH(0)
) fixed_rounding_inst(
    .data_in(data_out_unrounded), 
    .data_out(data_out) 
);

endmodule








// logic [2*IN_WIDTH-1:0] data_quantize [IN_PARALLELISM * IN_SIZE-1: 0];//data type

// for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: MAX
// {
//     if (data_in[i] > max_num) begin
//     max_num = data_in[i];
//     end
//     else begin
//         max_num = max_num;
//     end
// }
// end


// data_quantize[i]=data_in[i] / max_num;
// data_out[i]=data_quantize[i] << OUT_WIDTH;