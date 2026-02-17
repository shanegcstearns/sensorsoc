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
    input  wire                     cfg_hp_en_i,         // enable baseline
    input  wire [3:0]               cfg_ewma_shift_i,

    input  wire [DYN_W-1:0]         cfg_th_hi_i,         // burst enter threshold
    input  wire [DYN_W-1:0]         cfg_th_lo_i,         // burst exit threshold
    input  wire [DYN_W-1:0]         cfg_still_th_i,      // stillness threshold
    input  wire [7:0]               cfg_debounce_n_i,    // consecutive samples above TH_HI to enter burst

    input  wire [15:0]              cfg_epoch_len_i,     // samples per epoch (0 treated as 1)
    input  wire                     cfg_energy_sq_i,     // 0: dyn, 1: dyn^2

    // Per sample outputs
    output reg                      motion_valid_o,
    output reg  [MAG_W-1:0]         motion_mag_o,
    output reg  [DYN_W-1:0]         motion_dyn_o,
    output reg  [ENERGY_W-1:0]      motion_energy_accum_o,
    output reg                      burst_pulse_o,
    output reg                      in_burst_o,
    output reg                      stillness_flag_o,

    // Per epoch outputs (set on epoch done)
    output reg                      epoch_done_o,
    output reg  [ENERGY_W-1:0]      motion_energy_epoch_o,
    output reg  signed [ENERGY_W:0] motion_delta_epoch_o,
    output reg  [15:0]              burst_count_epoch_o,
    output reg  [15:0]              stillness_count_epoch_o,
    output reg  [15:0]              sample_count_epoch_o
);

    wire [15:0] epoch_len_eff = (cfg_epoch_len_i == 16'd0) ? 16'd1 : cfg_epoch_len_i;
    wire [7:0]  debounce_eff  = (cfg_debounce_n_i == 8'd0) ? 8'd1 : cfg_debounce_n_i;

    // Latch raw samples on posedge clk
    reg signed [AX_W-1:0] ax_r, ay_r, az_r;
    always @(posedge clk) begin
        if (!resetn) begin
            ax_r <= '0;
            ay_r <= '0;
            az_r <= '0;
        end else if (sample_valid_i && sample_ok_i) begin
            ax_r <= ax_i;
            ay_r <= ay_i;
            az_r <= az_i;
        end
    end

    // getting absolute values of each axis sample
    function automatic [AX_W-1:0] abs_s;
        input signed [AX_W-1:0] v;
        begin
            abs_s = v[AX_W-1] ? (~v + {{(AX_W-1){1'b0}}, 1'b1}) : v;
        end
    endfunction

    wire [AX_W-1:0] abs_ax = abs_s(ax_r);
    wire [AX_W-1:0] abs_ay = abs_s(ay_r);
    wire [AX_W-1:0] abs_az = abs_s(az_r);

    // getting combined magniute of all 3 axes
    wire [MAG_W-1:0] mag1_w = {{(MAG_W-AX_W){1'b0}}, abs_ax}
                            + {{(MAG_W-AX_W){1'b0}}, abs_ay}
                            + {{(MAG_W-AX_W){1'b0}}, abs_az};

    // Baseline EWMA and dynamic component
    reg [MAG_W-1:0] base_r;
    reg             base_init_r;

    wire signed [MAG_W:0] mag_ext  = $signed({1'b0, mag1_w});
    wire signed [MAG_W:0] base_ext = $signed({1'b0, base_r});
    wire signed [MAG_W:0] diff_w   = mag_ext - base_ext;
    wire signed [MAG_W:0] upd_w    = diff_w >>> cfg_ewma_shift_i;

    wire signed [MAG_W+1:0] base_next_signed = $signed({1'b0, base_r}) + $signed({1'b0, upd_w});
    wire [MAG_W-1:0] base_next_sat =
        base_next_signed[MAG_W+1] ? {MAG_W{1'b0}} :
        (base_next_signed > $signed({1'b0, {MAG_W{1'b1}}})) ? {MAG_W{1'b1}} :
        base_next_signed[MAG_W-1:0];

    wire signed [MAG_W:0] dyn_diff_w = mag_ext - $signed({1'b0, base_r});
    wire [DYN_W-1:0] dyn_hp_w = dyn_diff_w[MAG_W] ? {{(DYN_W-(MAG_W+1)){1'b0}}, (~dyn_diff_w + 1'b1)} :
                                            {{(DYN_W-(MAG_W+1)){1'b0}}, dyn_diff_w};

    wire [DYN_W-1:0] dyn_nohp_w = {{(DYN_W-MAG_W){1'b0}}, mag1_w};
    wire [DYN_W-1:0] dyn_sel_w  = cfg_hp_en_i ? dyn_hp_w : dyn_nohp_w;

    // Energy increment
    wire [ENERGY_W-1:0] dyn_ext = {{(ENERGY_W-DYN_W){1'b0}}, dyn_sel_w};
    wire [2*DYN_W-1:0]  dyn_sq_w = dyn_sel_w * dyn_sel_w;
    wire [ENERGY_W-1:0] dyn_sq_ext = (2*DYN_W >= ENERGY_W) ? dyn_sq_w[ENERGY_W-1:0]
                                                           : {{(ENERGY_W-2*DYN_W){1'b0}}, dyn_sq_w};
    wire [ENERGY_W-1:0] energy_add_w = cfg_energy_sq_i ? dyn_sq_ext : dyn_ext;

    // Burst detector
    reg [7:0]  hi_cnt_r;
    reg [15:0] burst_count_r;

    wire dyn_gt_hi = (dyn_sel_w > cfg_th_hi_i);
    wire dyn_lt_lo = (dyn_sel_w < cfg_th_lo_i);

    // Stillness per sample + epoch counter
    reg [15:0] still_cnt_r;
    wire still_w = (dyn_sel_w < cfg_still_th_i);

    wire will_enter_burst_w = (!in_burst_o) && dyn_gt_hi && (hi_cnt_r >= (debounce_eff - 8'd1));

    // Epoch counters and previous epoch for delta
    reg [15:0]         epoch_cnt_r;
    reg [ENERGY_W-1:0] prev_epoch_energy_r;

    always @(posedge clk) begin
        if (!resetn) begin
            motion_valid_o          <= 1'b0;
            motion_mag_o            <= {MAG_W{1'b0}};
            motion_dyn_o            <= {DYN_W{1'b0}};
            motion_energy_accum_o   <= {ENERGY_W{1'b0}};
            burst_pulse_o           <= 1'b0;
            in_burst_o              <= 1'b0;
            stillness_flag_o        <= 1'b0;

            base_r                  <= {MAG_W{1'b0}};
            base_init_r             <= 1'b0;

            hi_cnt_r                <= 8'd0;
            burst_count_r           <= 16'd0;
            still_cnt_r             <= 16'd0;

            epoch_cnt_r             <= 16'd0;
            prev_epoch_energy_r     <= {ENERGY_W{1'b0}};

            epoch_done_o            <= 1'b0;
            motion_energy_epoch_o   <= {ENERGY_W{1'b0}};
            motion_delta_epoch_o    <= '0;
            burst_count_epoch_o     <= 16'd0;
            stillness_count_epoch_o <= 16'd0;
            sample_count_epoch_o    <= 16'd0;
        end else begin
            motion_valid_o <= sample_fire;
            burst_pulse_o  <= 1'b0;
            epoch_done_o   <= 1'b0;

            if (sample_fire) begin
                motion_mag_o <= mag1_w;

                // Baseline update and dyn computation
                if (cfg_hp_en_i) begin
                    if (!base_init_r) begin
                        base_r      <= mag1_w;
                        base_init_r <= 1'b1;
                    end else begin
                        base_r <= base_next_sat;
                    end
                    motion_dyn_o <= dyn_hp_w;
                end else begin
                    base_r        <= base_r;
                    base_init_r   <= base_init_r;
                    motion_dyn_o  <= dyn_nohp_w;
                end

                // Stillness
                stillness_flag_o <= still_w;
                if (still_w) still_cnt_r <= still_cnt_r + 16'd1;

                // Burst logic with debounce
                if (!in_burst_o) begin
                    if (dyn_gt_hi) begin
                        if (hi_cnt_r != debounce_eff) hi_cnt_r <= hi_cnt_r + 8'd1;
                    end else begin
                        hi_cnt_r <= 8'd0;
                    end

                    if (will_enter_burst_w) begin
                        in_burst_o    <= 1'b1;
                        burst_pulse_o <= 1'b1;
                        burst_count_r <= burst_count_r + 16'd1;
                        hi_cnt_r      <= 8'd0;
                    end
                end else begin
                    hi_cnt_r <= 8'd0;
                    if (dyn_lt_lo) begin
                        in_burst_o <= 1'b0;
                    end
                end

                // Energy accumulation
                motion_energy_accum_o <= motion_energy_accum_o + energy_add_w;

                // Epoch handling (counts valid samples)
                epoch_cnt_r <= epoch_cnt_r + 16'd1;

                if (epoch_cnt_r == (epoch_len_eff - 16'd1)) begin
                    epoch_done_o            <= 1'b1;
                    motion_energy_epoch_o   <= motion_energy_accum_o + energy_add_w;
                    motion_delta_epoch_o    <= $signed({1'b0, (motion_energy_accum_o + energy_add_w)}) -
                                               $signed({1'b0, prev_epoch_energy_r});
                    prev_epoch_energy_r     <= motion_energy_accum_o + energy_add_w;
                    burst_count_epoch_o     <= burst_count_r + (will_enter_burst_w ? 16'd1 : 16'd0);
                    stillness_count_epoch_o <= still_cnt_r + (still_w ? 16'd1 : 16'd0);
                    sample_count_epoch_o    <= epoch_cnt_r + 16'd1;

                    // reset epoch accumulators for next epoch
                    motion_energy_accum_o   <= {ENERGY_W{1'b0}};
                    epoch_cnt_r             <= 16'd0;
                    burst_count_r           <= 16'd0;
                    still_cnt_r             <= 16'd0;
                end
            end
        end
    end

endmodule
