// 8-bit counter with synchronous reset and count enable.
module counter (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg  [7:0] count
);
    always @(posedge clk)
        if (rst) count <= 8'd0;
        else if (en) count <= count + 8'd1;
endmodule
