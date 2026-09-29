// 8-bit 4:1 multiplexer.
module mux (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire [7:0] c,
    input  wire [7:0] d,
    input  wire [1:0] sel,
    output wire [7:0] y
);
    assign y = sel[1] ? (sel[0] ? d : c) : (sel[0] ? b : a);
endmodule
