// 8-bit ALU: op picks add, sub, and, or, xor, shift left, shift right, less-than.
module alu (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire [2:0] op,
    output reg  [7:0] y,
    output wire       zero
);
    always @*
        case (op)
            3'd0: y = a + b;
            3'd1: y = a - b;
            3'd2: y = a & b;
            3'd3: y = a | b;
            3'd4: y = a ^ b;
            3'd5: y = a << b[2:0];
            3'd6: y = a >> b[2:0];
            default: y = {7'd0, a < b};
        endcase
    assign zero = (y == 8'd0);
endmodule
