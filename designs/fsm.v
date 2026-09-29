// Detects the bit sequence 1101 on `in`, overlapping. hit is high in the cycle the
// fourth bit arrives (a Mealy output); state is exposed so the compare sees it.
module fsm (
    input  wire       clk,
    input  wire       rst,
    input  wire       in,
    output wire       hit,
    output reg  [1:0] state
);
    localparam S0 = 2'd0, S1 = 2'd1, S11 = 2'd2, S110 = 2'd3;
    reg [1:0] next;
    always @*
        case (state)
            S0:      next = in ? S1 : S0;
            S1:      next = in ? S11 : S0;
            S11:     next = in ? S11 : S110;
            default: next = in ? S1 : S0;  // S110: 1101 complete, its last 1 may start a new one
        endcase
    always @(posedge clk)
        if (rst) state <= S0;
        else state <= next;
    assign hit = (state == S110) && in;
endmodule
