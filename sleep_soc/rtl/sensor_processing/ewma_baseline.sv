`timescale 1ns/1ps

module ewma_baseline #(
    parameter integer W = 16,
    parameter integer SHIFT_W = 4
) (
    input  wire                     clk_i,
    input  wire                     rst_ni,
    input  wire signed [W-1:0]      x_in_i,
    input  wire                     x_valid_i,
    input  wire                     baseline_en_i,
    input  wire [SHIFT_W-1:0]       alpha_shift_i,
    output reg  signed [W-1:0]      baseline_o,
    output reg                      baseline_valid_o
);

    always @(posedge clk_i or negedge rst_ni) begin
        reg signed [W:0] diff_v;
        reg signed [W:0] step_v;
        reg signed [W:0] next_v;
        if (!rst_ni) begin
            baseline_o <= '0;
            baseline_valid_o <= 1'b0;
        end else if (x_valid_i && baseline_en_i) begin
            if (!baseline_valid_o) begin
                baseline_o <= x_in_i;
                baseline_valid_o <= 1'b1;
            end else begin
                diff_v = $signed({x_in_i[W-1], x_in_i}) - $signed({baseline_o[W-1], baseline_o});
                step_v = diff_v >>> alpha_shift_i;
                next_v = $signed({baseline_o[W-1], baseline_o}) + step_v;
                baseline_o <= next_v[W-1:0];
            end
        end
    end

endmodule
