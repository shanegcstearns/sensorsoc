module feature_engine (
    input  wire                     clk_i,
    input  wire                     rst_ni,

    // enable on epoch end at top level
    input  wire                     enable_i,

    // Time feature inputs
    input  wire                     seconds_valid_i,
    input  wire [15:0]              cos_time_feat_o,

    // motion
    input wire                     motion_valid_i,
    input wire [15:0]              motion_energy_epoch_i

    // delta hr
    input wire                     delta_hr_valid_i,
    input wire signed [15:0]       delta_hr_i

    // rmssd
    input wire                     rmssd_valid_i,
    input wire signed [15:0]       rmssd_i

    // valid
    output reg                      feat_valid_o,
    output reg signed [15:0]        time_feat_o,
    output reg signed [15:0]        motion_feat_o,
    output reg signed [15:0]        delta_hr_feat_o,
    output reg signed [15:0]        rmssd_feat_o
);
    always @(posedge clk_i) begin
        if(!rst_ni) begin
            feat_valid_o <= 1'b0;
            time_feat_o <= 16'sd0;
            motion_feat_o <= 16'sd0;
            delta_hr_feat_o <= 16'sd0;
            rmssd_feat_o <= 16'sd0;
        end else if (enable_i && seconds_valid_i && motion_valid_i && delta_hr_valid_i && rmssd_valid_i) begin
            feat_valid_o <= 1'b1;
            time_feat_o <= cos_time_feat_o;
            motion_feat_o <= motion_energy_epoch_i; 
            delta_hr_feat_o <= delta_hr_i; 
            rmssd_feat_o <= rmssd_i;
        end else begin
            feat_valid_o <= 1'b0;
            time_feat_o <= 16'sd0;
            motion_feat_o <= 16'sd0;
            delta_hr_feat_o <= 16'sd0;
            rmssd_feat_o <= 16'sd0;
        end
    end

endmodule
