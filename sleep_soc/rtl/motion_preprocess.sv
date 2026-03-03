`timescale 1ns/1ps

module motion_preprocess #(
    parameter integer AX_W = 14,
    parameter integer MAG_W = (AX_W + 2),
    parameter integer DYN_W = (MAG_W + 1),
    parameter integer ENERGY_W = 48
)(
    input  wire                     clk,
    input  wire                     resetn,

    // Sample capture
    input  wire                     sample_valid_i,
    input  wire                     sample_ok_i,
    input  wire signed [AX_W-1:0]    ax_i,
    input  wire signed [AX_W-1:0]    ay_i,
    input  wire signed [AX_W-1:0]    az_i,

    // Config
    input  wire                     cfg_energy_sq_i,     // 0: mag, 1: mag^2

    // Epoch control (external)
    input  wire                     epoch_end_i,

    // Per epoch outputs (latched on epoch end)
    output reg                      epoch_done_o,
    output reg  [ENERGY_W-1:0]      motion_energy_epoch_o
);

    wire sample_fire = sample_valid_i && sample_ok_i;

    // absolute value per axis
    function automatic [AX_W-1:0] abs_s;
        input signed [AX_W-1:0] v;
        begin
            abs_s = v[AX_W-1] ? (~v + {{(AX_W-1){1'b0}}, 1'b1}) : v;
        end
    endfunction

    wire signed [AX_W-1:0] ax_s = sample_fire ? ax_i : {AX_W{1'b0}};
    wire signed [AX_W-1:0] ay_s = sample_fire ? ay_i : {AX_W{1'b0}};
    wire signed [AX_W-1:0] az_s = sample_fire ? az_i : {AX_W{1'b0}};

    wire [AX_W-1:0] abs_ax = abs_s(ax_s);
    wire [AX_W-1:0] abs_ay = abs_s(ay_s);
    wire [AX_W-1:0] abs_az = abs_s(az_s);

    // magnitude = |ax| + |ay| + |az|
    wire [MAG_W-1:0] mag1_w = {{(MAG_W-AX_W){1'b0}}, abs_ax}
                            + {{(MAG_W-AX_W){1'b0}}, abs_ay}
                            + {{(MAG_W-AX_W){1'b0}}, abs_az};

    // energy increment
    wire [ENERGY_W-1:0] mag_ext = {{(ENERGY_W-MAG_W){1'b0}}, mag1_w};
    wire [2*MAG_W-1:0]  mag_sq_w = mag1_w * mag1_w;
    wire [ENERGY_W-1:0] mag_sq_ext = (2*MAG_W >= ENERGY_W) ? mag_sq_w[ENERGY_W-1:0]
                                                           : {{(ENERGY_W-2*MAG_W){1'b0}}, mag_sq_w};
    wire [ENERGY_W-1:0] energy_add_w = cfg_energy_sq_i ? mag_sq_ext : mag_ext;

    reg [ENERGY_W-1:0] motion_energy_accum_r;
    wire [ENERGY_W-1:0] epoch_energy_w = motion_energy_accum_r + (sample_fire ? energy_add_w : {ENERGY_W{1'b0}});

    always @(posedge clk) begin
        if (!resetn) begin
            motion_energy_accum_r <= {ENERGY_W{1'b0}};
            motion_energy_epoch_o <= {ENERGY_W{1'b0}};
            epoch_done_o          <= 1'b0;
        end else begin
            epoch_done_o <= 1'b0;

            if (sample_fire) begin
                motion_energy_accum_r <= motion_energy_accum_r + energy_add_w;
            end

            if (epoch_end_i) begin
                motion_energy_epoch_o <= epoch_energy_w;
                epoch_done_o          <= 1'b1;
                motion_energy_accum_r <= {ENERGY_W{1'b0}};
            end
        end
    end

endmodule
