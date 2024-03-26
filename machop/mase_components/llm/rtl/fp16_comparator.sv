`timescale 1ns / 1ps


// for floating-point 16
// module fp16_comparator # (
//     parameter THRES = 30
// )(
//     input [16-1: 0] data_in,  // in fp-16 format
//     // S | E  E  E  E  E  | M M M ... M
//     // 15| 14 13 12 11 10 | 9 8 7 ... 0
    
//     output result
// );
//     logic [5-1: 0] exponent_biased;
//     logic sign;

//     assign sign = data_in[15];
//     assign exponent_biased = data_in[14:10];  // biased by 15. i.e., exponent_biased = exponenet + 15;
//     assign result = exponent_biased[4] && (exponent_biased[3] || exponent_biased[2] || exponent_biased[1]);  // exponent_biase >= 18, i.e., exponent >= 3
// endmodule

// for fixed-point 16
module fp16_comparator # (
    parameter THRES = 30  // currently deprecated
)(
    input [16-1: 0] data_in,  // by default ap_int<16>
    output result
);
    logic sign;
    assign sign = data_in[15];
    logic result_reg;

    always_comb begin
        if (sign) begin // negative
            result_reg = (!(& data_in[14:7])) || (data_in[7]); // -128 is treated as large number //(!data_in[14]) || (!data_in[13]);
        end else begin  // positive
            result_reg = | data_in[14:7];
        end
    end

    assign result = result_reg;
endmodule