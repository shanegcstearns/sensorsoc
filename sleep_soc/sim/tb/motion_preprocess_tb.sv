`timescale 1ns/1ps

module motion_preprocess_tb;

  localparam int AX_W = 14;

  logic clk = 0;
  logic resetn = 0;

  // sample interface
  logic sample_valid;
  logic sample_ok;
  logic signed [AX_W-1:0] ax, ay, az;

  // config
  logic        cfg_hp_en;
  logic [3:0]  cfg_ewma_shift;
  logic [16:0] cfg_th_hi;
  logic [16:0] cfg_th_lo;
  logic [16:0] cfg_still_th;
  logic [7:0]  cfg_debounce_n;
  logic [15:0] cfg_epoch_len;
  logic        cfg_energy_sq;

  // outputs
  wire        motion_valid;
  wire [15:0] motion_mag;
  wire [16:0] motion_dyn;
  wire [47:0] motion_energy_accum;
  wire        burst_pulse;
  wire        in_burst;
  wire        stillness_flag;

  wire        epoch_done;
  wire [47:0] motion_energy_epoch;
  wire signed [48:0] motion_delta_epoch;
  wire [15:0] burst_count_epoch;
  wire [15:0] stillness_count_epoch;
  wire [15:0] sample_count_epoch;

  motion_preprocess dut (
    .clk(clk),
    .resetn(resetn),

    .sample_valid_i(sample_valid),
    .sample_ok_i(sample_ok),
    .ax_i(ax),
    .ay_i(ay),
    .az_i(az),

    .cfg_hp_en_i(cfg_hp_en),
    .cfg_ewma_shift_i(cfg_ewma_shift),
    .cfg_th_hi_i(cfg_th_hi),
    .cfg_th_lo_i(cfg_th_lo),
    .cfg_still_th_i(cfg_still_th),
    .cfg_debounce_n_i(cfg_debounce_n),
    .cfg_epoch_len_i(cfg_epoch_len),
    .cfg_energy_sq_i(cfg_energy_sq),

    .motion_valid_o(motion_valid),
    .motion_mag_o(motion_mag),
    .motion_dyn_o(motion_dyn),
    .motion_energy_accum_o(motion_energy_accum),
    .burst_pulse_o(burst_pulse),
    .in_burst_o(in_burst),
    .stillness_flag_o(stillness_flag),

    .epoch_done_o(epoch_done),
    .motion_energy_epoch_o(motion_energy_epoch),
    .motion_delta_epoch_o(motion_delta_epoch),
    .burst_count_epoch_o(burst_count_epoch),
    .stillness_count_epoch_o(stillness_count_epoch),
    .sample_count_epoch_o(sample_count_epoch)
  );

  // 50 MHz
  always #10 clk = ~clk;

  task automatic send_sample(input int sax, input int say, input int saz);
    begin
      // Drive on negedge to avoid race with DUT posedge flops.
      @(negedge clk);
      ax = sax;
      ay = say;
      az = saz;
      sample_ok = 1'b1;
      sample_valid = 1'b1;
      @(negedge clk);
      sample_valid = 1'b0;
    end
  endtask

  int burst_pulses_seen;

  always @(posedge clk) begin
    if (!resetn) burst_pulses_seen <= 0;
    else if (burst_pulse) burst_pulses_seen <= burst_pulses_seen + 1;
  end

  logic epoch_done_seen;
  logic [47:0] cap_motion_energy_epoch;
  logic [15:0] cap_burst_count_epoch;
  logic [15:0] cap_stillness_count_epoch;
  always @(posedge clk) begin
    if (!resetn) begin
      epoch_done_seen <= 1'b0;
      cap_motion_energy_epoch <= '0;
      cap_burst_count_epoch <= '0;
      cap_stillness_count_epoch <= '0;
    end else if (epoch_done) begin
      epoch_done_seen <= 1'b1;
      cap_motion_energy_epoch <= motion_energy_epoch;
      cap_burst_count_epoch <= burst_count_epoch;
      cap_stillness_count_epoch <= stillness_count_epoch;
    end
  end

  initial begin
    sample_valid = 0;
    sample_ok    = 0;
    ax = '0; ay = '0; az = '0;

    cfg_hp_en       = 0;
    cfg_ewma_shift  = 4;
    cfg_th_hi       = 17'd1000;
    cfg_th_lo       = 17'd500;
    cfg_still_th    = 17'd50;
    cfg_debounce_n  = 8'd2;
    cfg_epoch_len   = 16'd16;
    cfg_energy_sq   = 1'b0;

    resetn = 0;
    repeat (10) @(posedge clk);
    resetn = 1;

    // A) steady still: no bursts, energy ~0, stillness count == epoch_len
    burst_pulses_seen = 0;
    epoch_done_seen = 0;
    repeat (16) send_sample(0, 0, 0);
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 0) $fatal(1, "steady still: unexpected burst pulses=%0d", burst_pulses_seen);
    if (!epoch_done_seen) $fatal(1, "steady still: did not observe epoch_done");
    if (cap_burst_count_epoch != 0) $fatal(1, "steady still: burst_count_epoch=%0d", cap_burst_count_epoch);
    if (cap_motion_energy_epoch != 0) $fatal(1, "steady still: motion_energy_epoch=%0d", cap_motion_energy_epoch);
    if (cap_stillness_count_epoch != cfg_epoch_len) $fatal(1, "steady still: stillness_count_epoch=%0d", cap_stillness_count_epoch);

    // B) single spike with higher debounce: should not enter burst
    cfg_debounce_n = 8'd3;
    burst_pulses_seen = 0;
    send_sample(2000, 0, 0);
    repeat (5) send_sample(0, 0, 0);
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 0) $fatal(1, "single spike: debounce failed, pulses=%0d", burst_pulses_seen);

    // C) periodic movement: expect 2 bursts (enter events) over a short run
    cfg_debounce_n = 8'd2;
    cfg_th_hi      = 17'd1200;
    cfg_th_lo      = 17'd600;
    burst_pulses_seen = 0;
    repeat (4) send_sample(0, 0, 0);
    repeat (5) send_sample(1500, 0, 0); // enter + stay
    repeat (4) send_sample(0, 0, 0);    // exit
    repeat (3) send_sample(1500, 0, 0); // enter again
    repeat (4) send_sample(0, 0, 0);
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 2) $fatal(1, "periodic: expected 2 burst pulses, got %0d", burst_pulses_seen);

    // D) slow "gravity-only" change with baseline removal enabled: no bursts
    cfg_hp_en      = 1'b1;
    cfg_ewma_shift = 2;        // fast baseline tracking
    cfg_th_hi      = 17'd500;
    cfg_th_lo      = 17'd250;
    cfg_debounce_n = 8'd2;
    burst_pulses_seen = 0;
    for (int i = 0; i < 64; i++) begin
      send_sample(i*64, 0, 0); // slow ramp
    end
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 0) $fatal(1, "baseline removal: unexpected bursts=%0d", burst_pulses_seen);

    $display("PASS");
    $finish;
  end

  initial begin
    $dumpfile("motion_preprocess_tb.vcd");
    $dumpvars(0, motion_preprocess_tb);
  end

endmodule
