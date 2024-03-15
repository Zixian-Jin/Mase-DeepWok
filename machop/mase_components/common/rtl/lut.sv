/*
Module      : lut
Description : This module implements a lookup table that is optionally
              output registered.
*/

`timescale 1ns/1ps

module lut #(
    parameter DATA_WIDTH = 8,
    parameter SIZE = 13,

    // Generate output register
    parameter OUTPUT_REG = 0,

    // Memfile
    parameter string MEM_FILE = "",

    // Derived params
    localparam ADDR_WIDTH = SIZE == 1 ? 1 : $clog2(SIZE)
) (
    input  logic                  clk,
    input  logic [ADDR_WIDTH-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

initial begin
    assert (SIZE > 0 && DATA_WIDTH > 0);
end

initial $readmemb(MEM_FILE, tab);

logic [DATA_WIDTH-1:0] tab [SIZE-1:0];

logic [DATA_WIDTH-1:0] look_up;
assign look_up = tab[addr];

generate
if (OUTPUT_REG) begin
    always_ff @(posedge clk) begin
        data <= look_up;
    end
end else begin
    assign data = look_up;
end
endgenerate

endmodule
