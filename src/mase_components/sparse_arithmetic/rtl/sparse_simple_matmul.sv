/*
Module      : sparse_simple_matmul
Description : This module does a matrix multiplcation between matrices X & Y.

              The dimensions for the matrix multiplcation are:
              n x m * m x k

              or in MASE naming convention
              a_dim1 x a_dim0 * b_dim1 x bdim_0

              Python equivalent:
              out = np.matmul(X, Y)
Reference   : simple_matmul from Derrick
*/

`timescale 1ns / 1ps

module sparse_simple_matmul #(
    // Dimensions
    parameter N                    = 2,
    parameter M                    = 2,
    parameter K                    = 2,
    // Input fixed point widths
    parameter X_WIDTH              = 8,
    parameter X_FRAC_WIDTH         = 1,
    parameter Y_WIDTH              = 8,
    parameter Y_FRAC_WIDTH         = 1,
    // Output fixed point widths
    // if OUTPUT_ROUNDING == 0:
    // then out_width & out_frac_width must match accumulator widths
    parameter OUTPUT_ROUNDING      = 1,
    parameter OUT_WIDTH            = 16,
    parameter OUT_FRAC_WIDTH       = 0,

    parameter BLOCK_NUM            = 2,   
    parameter SPARSE_BLOCK_NUM     = 1
) (
    input  logic                 clk,
    input  logic                 rst,

    // Input matrix X, row-wise ordering
    input  logic [X_WIDTH-1:0]   x_data [N*M-1:0],
    input  logic                 x_valid,
    output logic                 x_ready,

    // Input matrix Y, column-wise ordering
    input  logic [Y_WIDTH-1:0]   y_data [M*K-1:0],
    input  logic                 y_valid,
    output logic                 y_ready,

    // Output matrix
    output logic [OUT_WIDTH-1:0] out_data [N*K-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);



// Sparsity-related params
localparam BLOCK_SIZE = M/BLOCK_NUM;
localparam NONSPARSE_BLOCK_NUM = BLOCK_NUM - SPARSE_BLOCK_NUM;

initial begin
    assert (M % BLOCK_SIZE == 0) else
        $fatal("M is not divisible!");
end



// Accumulator widths in linear layer
localparam ACC_WIDTH = X_WIDTH + Y_WIDTH + $clog2(BLOCK_SIZE*NONSPARSE_BLOCK_NUM);
localparam ACC_FRAC_WIDTH = X_FRAC_WIDTH + Y_FRAC_WIDTH;

initial begin
    if(OUTPUT_ROUNDING == 0) begin
        assert (ACC_WIDTH == OUT_WIDTH) else
            $fatal("OUT_WIDTH must be %d if OUTPUT_ROUNDING == 0", ACC_WIDTH);
        assert (ACC_FRAC_WIDTH == OUT_FRAC_WIDTH) else
            $fatal("OUT_FRAC_WIDTH must be %d if OUTPUT_ROUNDING == 0",
                   ACC_FRAC_WIDTH);
    end
end



// N router_x
logic sync_x_valid;     // broadcast to N router_x
logic [N-1:0] sync_x_ready; 
// N*K router_y
logic sync_y_valid;    // broadcast to N*K router_y
logic [N*K-1:0] sync_y_ready;


logic inputs_valid, inputs_ready;
// NOTE: this `join2` module ensures x_data & y_data
// are fed in and processed in lockstep.
// TODO: alternatively, delete the join2 and split2 modules,
// put a join2 in EVERY router_y instances, which is 
// more hierarchical but consumes more resource.

join2 inputs_join (
    .data_in_valid ({x_valid, y_valid}),
    .data_in_ready ({x_ready, y_ready}),
    .data_out_valid(inputs_valid),
    .data_out_ready(inputs_ready)
);

split2 inputs_split (
    .data_in_valid (inputs_valid),
    .data_in_ready (inputs_ready),
    .data_out_valid ({sync_x_valid, sync_y_valid}),
    .data_out_ready ({&sync_x_ready, &sync_y_ready})
);



logic [N*K-1:0] fdp_out_ready;
logic [N*K-1:0] fdp_out_valid;


generate
for (genvar i = 0; i < N; i++) begin : multi_row

        // Slice a single row of x
        logic [X_WIDTH-1:0] row_x [M-1:0];
        assign row_x = x_data[(i+1)*M-1 : i*M];

        logic [BLOCK_NUM-1:0] nzc_flags;
        logic [X_WIDTH-1:0] active_row_x [BLOCK_SIZE*NONSPARSE_BLOCK_NUM-1:0]; 
        logic active_row_x_valid, active_row_x_ready;

        // Each `active_row_x` will be broadcast to `K` FDPs, 
        // each FDP uses one wire of `fdp_in_active_row_x_ready`
        logic [K-1:0] fdp_in_active_row_x_ready;
        assign active_row_x_ready = & fdp_in_active_row_x_ready;

        // check sparsity for BLOCK_NUM blocks of current row in parallel
        nzc_group #(
            .IN_WIDTH (X_WIDTH),
            .IN_SIZE (BLOCK_SIZE),
            .IN_PARALLELISM (BLOCK_NUM)  // BLOCK_SIZE*BLOCK_NUM = M
        ) row_nzc_inst (
            .data_in (row_x),
            .zero_flags (nzc_flags)
        );

        // route dense active_row_x from sparse row_x
        block_router #(
            .IN_BLOCK_NUM (BLOCK_NUM),
            .BLOCK_SIZE (BLOCK_SIZE),
            .OUT_BLOCK_NUM (NONSPARSE_BLOCK_NUM),
            .IN_WIDTH (X_WIDTH)
        ) router_row_x (
            .clk (clk),
            .rst (rst),
            .nonzero_sel (nzc_flags),
            .in_data (row_x),
            .in_valid (sync_x_valid),
            .in_ready (sync_x_ready[i]),
            .out_data (active_row_x),
            .out_valid (active_row_x_valid),
            .out_ready (active_row_x_ready)
        );

    for (genvar j = 0; j < K; j++) begin : multi_col
        // Slice a column of y
        logic [Y_WIDTH-1:0] col_y [M-1:0];
        for (genvar m = 0; m < M; m++) begin : col_assign
            assign col_y[m] = y_data[m*K+j];
        end

        logic [Y_WIDTH-1:0] active_col_y [BLOCK_SIZE*NONSPARSE_BLOCK_NUM-1:0];
        logic active_col_y_valid, active_col_y_ready;

        // route dense active_col_y from sparse col_y
        block_router #(
            .IN_BLOCK_NUM (BLOCK_NUM),
            .BLOCK_SIZE (BLOCK_SIZE),
            .OUT_BLOCK_NUM (NONSPARSE_BLOCK_NUM),
            .IN_WIDTH (Y_WIDTH)
        ) router_col_y (
            .clk (clk),
            .rst (rst),
            .nonzero_sel (nzc_flags),
            .in_data (col_y),
            .in_valid (sync_y_valid),
            .in_ready (sync_y_ready[i*K+j]),
            .out_data (active_col_y),
            .out_valid (active_col_y_valid),
            .out_ready (active_col_y_ready)
        );



        // Linear output
        logic [ACC_WIDTH-1:0] fdp_out_data;

        fixed_dot_product #(
            .IN_WIDTH             (X_WIDTH),
            .IN_SIZE              (BLOCK_SIZE*NONSPARSE_BLOCK_NUM),
            .WEIGHT_WIDTH         (Y_WIDTH)
        ) linear_inst (
            .clk                  (clk),
            .rst                  (rst),
            .data_in              (active_row_x),
            .data_in_valid        (active_row_x_valid),
            .data_in_ready        (fdp_in_active_row_x_ready[j]),
            .weight               (active_col_y),
            .weight_valid         (active_col_y_valid),
            /* verilator lint_off PINCONNECTEMPTY */
            // This pin is the same as data_in_ready pin
            .weight_ready         (active_col_y_ready),
            /* verilator lint_on PINCONNECTEMPTY */
            .data_out             (fdp_out_data),
            .data_out_valid       (fdp_out_valid[i*K+j]),
            .data_out_ready       (fdp_out_ready[i*K+j])
        );

        if (OUTPUT_ROUNDING) begin : rounding
            // Rounded output
            logic [OUT_WIDTH-1:0] rounded_dot_product;
            fixed_round #(
                .IN_WIDTH             (ACC_WIDTH),
                .IN_FRAC_WIDTH        (ACC_FRAC_WIDTH),
                .OUT_WIDTH            (OUT_WIDTH),
                .OUT_FRAC_WIDTH       (OUT_FRAC_WIDTH)
            ) round_inst (
                .data_in              (fdp_out_data),
                .data_out             (rounded_dot_product)
            );
            assign out_data[i*K+j] = rounded_dot_product;
        end else begin : no_rounding
            assign out_data[i*K+j] = fdp_out_data;
        end

    end
end
endgenerate


assign out_valid = &fdp_out_valid;
assign fdp_out_ready = {(N*K){out_ready}};

endmodule
