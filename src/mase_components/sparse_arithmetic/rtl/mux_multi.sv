`timescale 1ns / 1ps
/*
 * Module: mux_multi
 * Description: this module multiplexes IN_BLOCK_NUM input blocks to OUT_BLOCK_NUM output blocks
 *              multiplexing is controlled by non_zero_sel, where there are OUT_BLOCK_NUM bits of 0.
 */
module mux_multi #(
    parameter IN_BLOCK_NUM = 3,
    parameter BLOCK_SIZE = 4,
    parameter OUT_BLOCK_NUM = 2,
    parameter IN_WIDTH = 16
) (
    input [IN_BLOCK_NUM-1: 0] nonzero_sel,
    input [IN_WIDTH-1 :0] din [IN_BLOCK_NUM * BLOCK_SIZE -1 :0],
    output [IN_WIDTH-1 :0] dout [OUT_BLOCK_NUM * BLOCK_SIZE-1 :0]
);
    logic [IN_WIDTH-1 :0] reg_out [OUT_BLOCK_NUM * BLOCK_SIZE-1 :0];
    integer i, j, nonzero_block_id;

    always_comb begin

        for (int k = 0; k < OUT_BLOCK_NUM * BLOCK_SIZE; k++) begin
            reg_out[k] = '0;
        end

        i = 0;
        j = 0;
        nonzero_block_id = 0;
        for (int block_id = 0; block_id < IN_BLOCK_NUM; block_id++) begin
            if (nonzero_sel[block_id] == 1'b0) begin
                // non-zero block
                for (int k=0; k<BLOCK_SIZE; k++) begin
                    i = block_id*BLOCK_SIZE + k;
                    j = nonzero_block_id*BLOCK_SIZE + k;
                    reg_out[j] = din[i];
                end
                nonzero_block_id = nonzero_block_id + 1;
            end
        end
    end


    // for (genvar i = 0; i < IN_BLOCK_NUM; i++) begin
    //     always_latch begin
    //         if (nonzero_sel[i] == 1'b0) begin
    //             // reg_out[(j+1)*BLOCK_SIZE - 1 : j*BLOCK_SIZE] = din[(i+1)*BLOCK_SIZE - 1 : i*BLOCK_SIZE];
    //             reg_out[j] = din[i];
    //             j = j + 1;
    //         end
    //     end
    // end

    assign dout = reg_out;
endmodule