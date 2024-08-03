module matmul_top(
    input clk
);

    localparam N = 5;
    localparam NZN_ROW = 2;
    localparam NZN = N*NZN_ROW;  // 5 rows
    localparam X_WIDTH = 16;
    localparam ADDR_WIDTH = 16;
    
    logic [X_WIDTH-1:0]   x_data [NZN-1:0];
    logic [ADDR_WIDTH-1:0]   x_row_table [NZN-1:0];
    logic [ADDR_WIDTH-1:0]   x_col_table [NZN-1:0];
    
    wire [X_WIDTH-1:0] data_in_dense [NZN_ROW-1:0];
    wire [2*X_WIDTH-1:0] add_out_dense [NZN_ROW-1:0];
    wire [X_WIDTH-1:0] bias;

    logic [2*X_WIDTH-1:0] data_out_dense [NZN_ROW-1:0];
        
    // initialization
    initial begin
        for (int i = 0; i < NZN; i += 1) begin
            x_data[i] = 2;
            x_row_table[i] = 0;
            x_col_table[i] = 0;
        end
    end
    assign bias = 1;

    wire [X_WIDTH-1:0] data_in_dense [NZN_ROW-1:0];
    wire [ADDR_WIDTH-1:0] out_index [NZN_ROW-1:0];
    csr_row_fetcher #(
        .IN_SIZE (NZN),
        .FETCH_SIZE (NZN_ROW),
        .DATA_WIDTH (X_WIDTH),
        .IN_PARALLELISM (N),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) row_x_fetcher (
        .in_data (x_data),
        .in_index (x_col_table),
        .in_bounds (x_row_table[N-1:0]),
        .fetch_row (N-1-i),  // TODO
        .out_data (data_in_dense),
        .out_index (out_index)
    );
    
    logic [X_WIDTH-1:0] add_in_dense [NZN_ROW-1:0];
    
    for (genvar i = 0; i < NZN_ROW; i += 1) begin : trivial_addition
        assign add_out_dense[i] = add_in_dense[i] + bias; 
    end

    always_ff @(clk) begin : trival_seq
        add_in_dense <= data_in_dense;
        data_out_dense <= add_out_dense;
    end

endmodule