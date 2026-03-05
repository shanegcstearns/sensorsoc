`timescale 1ns/1ps

module top #(
    parameter int unsigned CLK_HZ = 10_000_000,
    parameter int unsigned GT_CLK_HZ = 10_000_000,
    parameter int unsigned GT_EPOCH_HZ = 100,
    parameter int unsigned GT_EPOCH_COUNT_MAX = 1000,

    parameter int unsigned ACC_POLL_PERIOD_TICKS = 50_000,
    parameter int unsigned PPG_POLL_PERIOD_TICKS = 100,
    parameter int unsigned PPG_WATERMARK = 8,
    parameter int unsigned PPG_MAX_BURST_SAMPLES = 32,

    parameter logic [31:0] CFG_REFRACT_MS = 32'd250,
    parameter logic [31:0] CFG_RR_MIN_MS = 32'd300,
    parameter logic [31:0] CFG_RR_MAX_MS = 32'd2000,

    parameter logic [7:0] CFG_Q_MIN_ACCEPT = 8'd10,

    parameter logic [7:0] CFG_BEAT_Q_MIN = 8'd16,
    parameter logic [7:0] CFG_MIN_VALID_FRAC = 8'd96,
    parameter logic [7:0] CFG_MAX_DOUBLE = 8'd4,
    parameter logic [7:0] CFG_MAX_MISSED = 8'd3,
    parameter logic [15:0] CFG_MOTION_HI_TH = 16'd2000,
    parameter logic [15:0] CFG_MAX_MOTION_HI = 16'd3,
    parameter logic [31:0] COS_PERIOD_SECONDS = 32'd86400,
    parameter logic [2:0]  COS_LUT_BITS = 3'd6,
    parameter logic [15:0] COS_SCALE_Q15 = 16'h7FFF,

    parameter int unsigned RMSSD_MIN_RR_COUNT = 1
) (
    input  logic clk_i,
    input  logic reset_i,

    // Functional simulation bus to sensor models (through i2c_master).
    output logic        sim_req_o,
    output logic [6:0]  sim_addr_o,
    output logic [7:0]  sim_reg_o,
    output logic [7:0]  sim_len_o,
    output logic        sim_write_o,
    output logic [7:0]  sim_wdata_o,
    input  logic        sim_ack_i,
    input  logic [7:0]  sim_rdata_i,
    input  logic        sim_rvalid_i,
    input  logic        sim_rlast_i,
    input  logic        sim_err_i,

    // Pipeline outputs toward ML
    output logic                      feat_valid_o,
    output logic signed [15:0]        time_feat_o,
    output logic signed [15:0]        motion_feat_o,
    output logic signed [15:0]        delta_hr_feat_o,
    output logic signed [15:0]        rmssd_feat_o,

    // Signal quality outputs
    output logic                      ml_update_gate_o,
    output logic [7:0]                invalid_reason_o,

    // Epoch pulse for TB orchestration
    output logic                      epoch_end_o,

    output logic                      alarm_o
);

    localparam logic [11:0] CFG_LP_BETA_Q10      = 12'd128;
    localparam logic [11:0] CFG_BASE_ALPHA_Q10   = 12'd16;
    localparam logic [23:0] CFG_ENV_DECAY        = 24'd8;
    localparam logic [11:0] CFG_THR_K_Q10        = 12'd512;
    localparam logic [23:0] CFG_THR_MIN          = 24'd32;
    localparam logic [7:0]  CFG_Q_AMP_W          = 8'd4;
    localparam logic [7:0]  CFG_Q_SLOPE_W        = 8'd2;
    localparam logic [7:0]  CFG_Q_REFRAC_PENALTY = 8'd24;

    localparam int unsigned MS_DIV = (CLK_HZ >= 1000) ? (CLK_HZ / 1000) : 1;
    localparam int unsigned MS_DIV_W = (MS_DIV <= 1) ? 1 : $clog2(MS_DIV);

    logic [MS_DIV_W-1:0] ms_div_q;
    logic [31:0]         time_ms_w;

    logic [15:0] seconds_w;
    logic epoch_end_w;
    logic signed [15:0] cos_time_w;

    logic signed [15:0] ax_w;
    logic signed [15:0] ay_w;
    logic signed [15:0] az_w;
    logic accel_valid_w;
    logic accel_error_w;
    logic [15:0] motion_inst_mag_w;

    logic motion_epoch_w;
    logic [47:0] motion_energy_w;

    logic [15:0] ppg_sample_w;
    logic ppg_sample_valid_w;
    logic [31:0] ppg_sample_time_w;
    logic fifo_overflow_w;
    logic fifo_empty_w;
    logic ppg_i2c_err_w;

    logic beat_pulse_w;
    logic rr_valid_w;
    logic rr_accepted_w;
    logic [31:0] rr_interval_w;
    logic signed [16:0] delta_hr_w;
    logic [7:0] beat_quality_w;
    logic double_beat_w;
    logic missed_beat_w;
    logic ppg_invalid_w;

    logic [31:0] rmssd_w;
    logic rmssd_valid_w;

    logic epoch_end_d;
    logic fifo_overflow_d;
    logic ppg_i2c_err_d;
    logic fifo_overflow_event_w;
    logic ppg_i2c_err_event_w;

    logic       acc_i2c_cmd_valid_w;
    logic       acc_i2c_cmd_ready_w;
    logic [6:0] acc_i2c_cmd_addr_w;
    logic [7:0] acc_i2c_cmd_reg_w;
    logic [7:0] acc_i2c_cmd_len_w;
    logic       acc_i2c_cmd_write_w;
    logic [7:0] acc_i2c_cmd_wdata_w;
    logic       acc_i2c_rsp_valid_w;
    logic [7:0] acc_i2c_rsp_data_w;
    logic       acc_i2c_rsp_done_w;
    logic       acc_i2c_rsp_error_w;

    logic       ppg_i2c_cmd_valid_w;
    logic       ppg_i2c_cmd_ready_w;
    logic [6:0] ppg_i2c_cmd_addr_w;
    logic [7:0] ppg_i2c_cmd_reg_w;
    logic [7:0] ppg_i2c_cmd_len_w;
    logic       ppg_i2c_cmd_write_w;
    logic [7:0] ppg_i2c_cmd_wdata_w;
    logic       ppg_i2c_rsp_valid_w;
    logic [7:0] ppg_i2c_rsp_data_w;
    logic       ppg_i2c_rsp_last_w;
    logic       ppg_i2c_rsp_done_w;
    logic       ppg_i2c_rsp_err_w;
    logic       ppg_i2c_rsp_ready_w;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            ms_div_q  <= '0;
            time_ms_w <= 32'd0;
            epoch_end_d <= 1'b0;
            fifo_overflow_d <= 1'b0;
            ppg_i2c_err_d <= 1'b0;
        end else begin
            if (ms_div_q == MS_DIV-1) begin
                ms_div_q  <= '0;
                time_ms_w <= time_ms_w + 32'd1;
            end else begin
                ms_div_q <= ms_div_q + 1'b1;
            end
            epoch_end_d <= epoch_end_w;
            fifo_overflow_d <= fifo_overflow_w;
            ppg_i2c_err_d <= ppg_i2c_err_w;
        end
    end

    // Convert sticky FIFO flags into edge events so one old error does not
    // permanently hold ML gating low across all future epochs.
    assign fifo_overflow_event_w = fifo_overflow_w & ~fifo_overflow_d;
    assign ppg_i2c_err_event_w = ppg_i2c_err_w & ~ppg_i2c_err_d;

    globaltimer #(
        .clk_speed_hz(GT_CLK_HZ),
        .epoch_hz(GT_EPOCH_HZ),
        .epoch_count_max(GT_EPOCH_COUNT_MAX)
    ) u_globaltimer (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .en_i(1'b1),
        .time_in_night_seconds_o(seconds_w),
        .epoch_end_o(epoch_end_w),
        .epoch_index_o()
    );

    cos_lut_timer u_cos (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .cfg_enable_i(1'b1),
        .seconds_in_night_i({16'h0000, seconds_w}),
        .seconds_valid_i(1'b1),
        .cfg_period_seconds_i(COS_PERIOD_SECONDS),
        .cfg_lut_bits_i(COS_LUT_BITS),
        .cfg_scale_q15_i(COS_SCALE_Q15),
        .cos_time_feat_o(cos_time_w)
    );

    i2c_master u_i2c_master (
        .clk(clk_i),
        .resetn(~reset_i),
        .accel_cmd_valid_i(acc_i2c_cmd_valid_w),
        .accel_cmd_ready_o(acc_i2c_cmd_ready_w),
        .accel_cmd_addr_i(acc_i2c_cmd_addr_w),
        .accel_cmd_reg_i(acc_i2c_cmd_reg_w),
        .accel_cmd_len_i(acc_i2c_cmd_len_w),
        .accel_cmd_write_i(acc_i2c_cmd_write_w),
        .accel_cmd_wdata_i(acc_i2c_cmd_wdata_w),
        .accel_rsp_valid_o(acc_i2c_rsp_valid_w),
        .accel_rsp_data_o(acc_i2c_rsp_data_w),
        .accel_rsp_last_o(),
        .accel_rsp_done_o(acc_i2c_rsp_done_w),
        .accel_rsp_err_o(acc_i2c_rsp_error_w),
        .accel_rsp_ready_i(1'b1),
        .ppg_cmd_valid_i(ppg_i2c_cmd_valid_w),
        .ppg_cmd_ready_o(ppg_i2c_cmd_ready_w),
        .ppg_cmd_addr_i(ppg_i2c_cmd_addr_w),
        .ppg_cmd_reg_i(ppg_i2c_cmd_reg_w),
        .ppg_cmd_len_i(ppg_i2c_cmd_len_w),
        .ppg_cmd_write_i(ppg_i2c_cmd_write_w),
        .ppg_cmd_wdata_i(ppg_i2c_cmd_wdata_w),
        .ppg_rsp_valid_o(ppg_i2c_rsp_valid_w),
        .ppg_rsp_data_o(ppg_i2c_rsp_data_w),
        .ppg_rsp_last_o(ppg_i2c_rsp_last_w),
        .ppg_rsp_done_o(ppg_i2c_rsp_done_w),
        .ppg_rsp_err_o(ppg_i2c_rsp_err_w),
        .ppg_rsp_ready_i(ppg_i2c_rsp_ready_w),
        .sim_req(sim_req_o),
        .sim_addr(sim_addr_o),
        .sim_reg(sim_reg_o),
        .sim_len(sim_len_o),
        .sim_write(sim_write_o),
        .sim_wdata(sim_wdata_o),
        .sim_ack(sim_ack_i),
        .sim_rdata(sim_rdata_i),
        .sim_rvalid(sim_rvalid_i),
        .sim_rlast(sim_rlast_i),
        .sim_err(sim_err_i)
    );

    accel_reader u_accel_reader (
        .clk(clk_i),
        .rst_i(reset_i),
        .cfg_enable_i(1'b1),
        .cfg_init_en_i(1'b1),
        .cfg_poll_period_ticks_i(ACC_POLL_PERIOD_TICKS),
        .cfg_ctrl1_data_i(8'h57),
        .cfg_range_data_i(8'h00),
        .i2c_cmd_valid_o(acc_i2c_cmd_valid_w),
        .i2c_cmd_ready_i(acc_i2c_cmd_ready_w),
        .i2c_cmd_addr_o(acc_i2c_cmd_addr_w),
        .i2c_cmd_reg_o(acc_i2c_cmd_reg_w),
        .i2c_cmd_len_o(acc_i2c_cmd_len_w),
        .i2c_cmd_write_o(acc_i2c_cmd_write_w),
        .i2c_cmd_wdata_o(acc_i2c_cmd_wdata_w),
        .i2c_rsp_valid_i(acc_i2c_rsp_valid_w),
        .i2c_rsp_data_i(acc_i2c_rsp_data_w),
        .i2c_rsp_done_i(acc_i2c_rsp_done_w),
        .i2c_rsp_error_i(acc_i2c_rsp_error_w),
        .ax_o(ax_w),
        .ay_o(ay_w),
        .az_o(az_w),
        .accel_valid_o(accel_valid_w),
        .init_done_o(),
        .i2c_error_o(accel_error_w),
        .timeout_o(),
        .nack_seen_o()
    );

    motion_preprocess #(
        .AX_W(16)
    ) u_motion_preprocess (
        .clk(clk_i),
        .rst_i(reset_i),
        .sample_valid_i(accel_valid_w),
        // accel_valid_o already indicates a completed good read.
        .sample_ok_i(1'b1),
        .ax_i(ax_w),
        .ay_i(ay_w),
        .az_i(az_w),
        .epoch_end_i(epoch_end_w),
        .epoch_done_o(motion_epoch_w),
        .motion_energy_epoch_o(motion_energy_w)
    );

    // Per-sample motion magnitude for signal-quality high-motion counting.
    assign motion_inst_mag_w =
        (ax_w[15] ? (~ax_w + 16'd1) : ax_w) +
        (ay_w[15] ? (~ay_w + 16'd1) : ay_w) +
        (az_w[15] ? (~az_w + 16'd1) : az_w);

    ppg_fifo_reader #(
        .POLL_PERIOD(PPG_POLL_PERIOD_TICKS),
        .WATERMARK(PPG_WATERMARK),
        .MAX_BURST_SAMPLES(PPG_MAX_BURST_SAMPLES)
    ) u_ppg_fifo_reader (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .t_now(time_ms_w),
        .i2c_cmd_valid(ppg_i2c_cmd_valid_w),
        .i2c_cmd_ready(ppg_i2c_cmd_ready_w),
        .i2c_cmd_addr(ppg_i2c_cmd_addr_w),
        .i2c_cmd_reg(ppg_i2c_cmd_reg_w),
        .i2c_cmd_len(ppg_i2c_cmd_len_w),
        .i2c_cmd_write(ppg_i2c_cmd_write_w),
        .i2c_cmd_wdata(ppg_i2c_cmd_wdata_w),
        .i2c_rsp_valid(ppg_i2c_rsp_valid_w),
        .i2c_rsp_data(ppg_i2c_rsp_data_w),
        .i2c_rsp_last(ppg_i2c_rsp_last_w),
        .i2c_rsp_err(ppg_i2c_rsp_err_w),
        .i2c_rsp_ready(ppg_i2c_rsp_ready_w),
        .ppg_sample(ppg_sample_w),
        .ppg_sample_valid(ppg_sample_valid_w),
        .ppg_sample_time(ppg_sample_time_w),
        .fifo_overflow_flag(fifo_overflow_w),
        .fifo_empty_flag(fifo_empty_w),
        .i2c_error_flag(ppg_i2c_err_w)
    );

    ppg_beat_detect_rr_calc u_beat_detect (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .ppg_sample_i(ppg_sample_w),
        .ppg_valid_i(ppg_sample_valid_w),
        .ppg_sample_time_i(ppg_sample_time_w),
        .cfg_enable_i(1'b1),
        .cfg_bypass_i(1'b0),
        .cfg_signed_i(1'b0),
        .cfg_lp_beta_i(CFG_LP_BETA_Q10),
        .cfg_base_alpha_i(CFG_BASE_ALPHA_Q10),
        .cfg_env_decay_i(CFG_ENV_DECAY),
        .cfg_abs_en_i(1'b1),
        .cfg_thr_k_i(CFG_THR_K_Q10),
        .cfg_thr_min_i(CFG_THR_MIN),
        .cfg_refrac_ticks_i(CFG_REFRACT_MS),
        .cfg_rr_min_ticks_i(CFG_RR_MIN_MS),
        .cfg_rr_max_ticks_i(CFG_RR_MAX_MS),
        .cfg_peak_mode_i(1'b0),
        .cfg_q_amp_w_i(CFG_Q_AMP_W),
        .cfg_q_slope_w_i(CFG_Q_SLOPE_W),
        .cfg_q_refrac_penalty_i(CFG_Q_REFRAC_PENALTY),
        .cfg_q_min_accept_i(CFG_Q_MIN_ACCEPT),
        .beat_pulse_o(beat_pulse_w),
        .rr_valid_o(rr_valid_w),
        .rr_accepted_o(rr_accepted_w),
        .rr_interval_o(rr_interval_w),
        .delta_hr_bpm_o(delta_hr_w),
        .beat_quality_o(beat_quality_w),
        .double_beat_o(double_beat_w),
        .missed_beat_o(missed_beat_w),
        .ppg_invalid_o(ppg_invalid_w)
    );

    rmssd_engine #(
        .MIN_RR_COUNT(RMSSD_MIN_RR_COUNT)
    ) u_rmssd (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .rr_interval_i(rr_interval_w),
        .rr_valid_i(rr_valid_w),
        .rr_accepted_i(rr_accepted_w),
        .epoch_end_i(epoch_end_w),
        .rmssd_epoch_o(rmssd_w),
        .rmssd_valid_o(rmssd_valid_w),
        .rr_diff_count_o()
    );

    signal_quality u_signal_quality (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .epoch_end_i(epoch_end_w),
        .beat_event_i(beat_pulse_w),
        .beat_quality_i(beat_quality_w),
        .double_beat_i(double_beat_w),
        .missed_beat_i(missed_beat_w),
        .fifo_overflow_i(fifo_overflow_event_w),
        .fifo_i2c_error_i(ppg_i2c_err_event_w),
        .motion_valid_i(accel_valid_w),
        .motion_intensity_i(motion_inst_mag_w),
        .cfg_beat_q_min_i(CFG_BEAT_Q_MIN),
        .cfg_min_valid_fraction_i(CFG_MIN_VALID_FRAC),
        .cfg_max_double_i(CFG_MAX_DOUBLE),
        .cfg_max_missed_i(CFG_MAX_MISSED),
        .cfg_motion_hi_th_i(CFG_MOTION_HI_TH),
        .cfg_max_motion_hi_i(CFG_MAX_MOTION_HI),
        .invalid_reason_o(invalid_reason_o),
        .ml_update_gate_o(ml_update_gate_o)
    );

    feature_engine u_feature_engine (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .enable_i(epoch_end_d),
        .seconds_valid_i(1'b1),
        .cos_time_feat_i(cos_time_w),
        .motion_valid_i(motion_epoch_w),
        .motion_energy_epoch_i(motion_energy_w[15:0]),
        .delta_hr_valid_i(rr_valid_w),
        .delta_hr_i(delta_hr_w[15:0]),
        .rmssd_valid_i(rmssd_valid_w),
        .rmssd_i(rmssd_w[15:0]),
        .feat_valid_o(feat_valid_o),
        .time_feat_o(time_feat_o),
        .motion_feat_o(motion_feat_o),
        .delta_hr_feat_o(delta_hr_feat_o),
        .rmssd_feat_o(rmssd_feat_o),
        .ml_update_gate_i(ml_update_gate_o)
    );

    assign epoch_end_o = epoch_end_w;
    assign alarm_o = 1'b0;

endmodule
