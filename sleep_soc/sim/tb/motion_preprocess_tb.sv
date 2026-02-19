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
  always #10 clk = about clk;

  task automatic send_sample(input int sax, input int say, input int saz, input bit ok);
    begin
      @(negedge clk);
      ax = sax;
      ay = say;
      az = saz;
      sample_ok = ok;
      sample_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      sample_valid = 1'b0;
      sample_ok = 1'b0;
    end
  endtask

  task automatic do_reset();
    begin
      resetn = 0;
      repeat (5) @(posedge clk);
      resetn = 1;
      repeat (2) @(posedge clk);
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
    bit still_seen;
    logic [16:0] dyn_first;
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

    do_reset();

    // test 1: sample ok gating + mag/dyn/energy
    cfg_hp_en      = 1'b0;
    cfg_energy_sq  = 1'b0;
    cfg_epoch_len  = 16'd4;
    cfg_th_hi      = 17'd2000;
    cfg_th_lo      = 17'd1500;
    cfg_still_th   = 17'd200;
    cfg_debounce_n = 8'd2;
    burst_pulses_seen = 0;
    epoch_done_seen = 0;
    send_sample(10, 0, 0, 1'b0);
    if (motion_valid) $fatal(1, "sample_ok gating: motion_valid asserted with sample_ok=0");
    send_sample(3, -4, 5, 1'b1); // prime
    send_sample(0, 0, 0, 1'b1);  // check previous sample
    if (!motion_valid) $fatal(1, "mag/dyn: motion_valid not asserted");
    if (motion_mag !== 16'd12) $fatal(1, "mag mismatch: got=%0d exp=12", motion_mag);
    if (motion_dyn !== 17'd12) $fatal(1, "dyn mismatch: got=%0d exp=12", motion_dyn);
    if (motion_energy_accum !== 48'd12) $fatal(1, "energy mismatch: got=%0d exp=12", motion_energy_accum);

    // test 2: epoch_len=0 treated as 1
    do_reset();
    cfg_hp_en      = 1'b0;
    cfg_energy_sq  = 1'b0;
    cfg_epoch_len  = 16'd0;
    cfg_th_hi      = 17'd1000;
    cfg_th_lo      = 17'd500;
    cfg_still_th   = 17'd1;
    cfg_debounce_n = 8'd2;
    epoch_done_seen = 0;
    send_sample(1, 0, 0, 1'b1);
    if (!epoch_done) $fatal(1, "epoch_len=0: epoch_done not asserted");
    if (sample_count_epoch != 16'd1) $fatal(1, "epoch_len=0: sample_count_epoch=%0d", sample_count_epoch);
    if (motion_energy_epoch != 48'd0) $fatal(1, "epoch_len=0: motion_energy_epoch=%0d", motion_energy_epoch);

    // test 3: energy_sq mode with epoch_len=1
    do_reset();
    cfg_hp_en      = 1'b0;
    cfg_energy_sq  = 1'b1;
    cfg_epoch_len  = 16'd1;
    cfg_th_hi      = 17'd1000;
    cfg_th_lo      = 17'd500;
    cfg_still_th   = 17'd1;
    cfg_debounce_n = 8'd2;
    send_sample(3, 0, 0, 1'b1); // prime
    send_sample(4, 0, 0, 1'b1); // check energy for previous sample
    if (!epoch_done) $fatal(1, "energy_sq: epoch_done not asserted");
    if (motion_energy_epoch != 48'd9) $fatal(1, "energy_sq: motion_energy_epoch=%0d", motion_energy_epoch);

    // test 4: debounce_n=0 treated as 1, burst enter/exit
    do_reset();
    cfg_hp_en      = 1'b0;
    cfg_energy_sq  = 1'b0;
    cfg_epoch_len  = 16'd8;
    cfg_th_hi      = 17'd10;
    cfg_th_lo      = 17'd5;
    cfg_still_th   = 17'd1;
    cfg_debounce_n = 8'd0;
    burst_pulses_seen = 0;
    send_sample(20, 0, 0, 1'b1); // prime high
    send_sample(0, 0, 0, 1'b1);  // should enter burst based on previous sample
    if (!burst_pulse) $fatal(1, "debounce=0: burst_pulse not asserted");
    if (!in_burst) $fatal(1, "debounce=0: in_burst not asserted");
    send_sample(0, 0, 0, 1'b1);  // should exit burst based on previous low
    if (in_burst) $fatal(1, "burst exit: in_burst not cleared");

    // test 5: steady still: no bursts, energy about 0, stillness count == epoch_len
    do_reset();
    cfg_hp_en       = 0;
    cfg_ewma_shift  = 4;
    cfg_th_hi       = 17'd1000;
    cfg_th_lo       = 17'd500;
    cfg_still_th    = 17'd50;
    cfg_debounce_n  = 8'd2;
    cfg_epoch_len   = 16'd16;
    cfg_energy_sq   = 1'b0;
    burst_pulses_seen = 0;
    epoch_done_seen = 0;
    repeat (16) send_sample(0, 0, 0, 1'b1);
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 0) $fatal(1, "steady still: unexpected burst pulses=%0d", burst_pulses_seen);
    if (!epoch_done_seen) $fatal(1, "steady still: did not observe epoch_done");
    if (cap_burst_count_epoch != 0) $fatal(1, "steady still: burst_count_epoch=%0d", cap_burst_count_epoch);
    if (cap_motion_energy_epoch != 0) $fatal(1, "steady still: motion_energy_epoch=%0d", cap_motion_energy_epoch);
    if (cap_stillness_count_epoch != cfg_epoch_len) $fatal(1, "steady still: stillness_count_epoch=%0d", cap_stillness_count_epoch);

    // test 6: single spike with higher debounce: should not enter burst
    cfg_debounce_n = 8'd3;
    burst_pulses_seen = 0;
    send_sample(2000, 0, 0, 1'b1);
    repeat (5) send_sample(0, 0, 0, 1'b1);
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 0) $fatal(1, "single spike: debounce failed, pulses=%0d", burst_pulses_seen);

    // test 7: periodic movement: expect 2 bursts over short run
    cfg_debounce_n = 8'd2;
    cfg_th_hi      = 17'd1200;
    cfg_th_lo      = 17'd600;
    burst_pulses_seen = 0;
    repeat (4) send_sample(0, 0, 0, 1'b1);
    repeat (5) send_sample(1500, 0, 0, 1'b1); // enter + stay
    repeat (4) send_sample(0, 0, 0, 1'b1);    // exit
    repeat (3) send_sample(1500, 0, 0, 1'b1); // enter again
    repeat (4) send_sample(0, 0, 0, 1'b1);
    repeat (2) @(posedge clk);
    if (burst_pulses_seen != 2) $fatal(1, "periodic: expected 2 burst pulses, got %0d", burst_pulses_seen);

    // test 8: baseline removal enabled: constant signal should settle to about 0 dyn
    do_reset();
    cfg_hp_en      = 1'b1;
    cfg_ewma_shift = 2;        // fast baseline tracking
    cfg_th_hi      = 17'd2000;
    cfg_th_lo      = 17'd1500;
    cfg_still_th   = 17'd5;
    cfg_debounce_n = 8'd2;
    burst_pulses_seen = 0;
    send_sample(1000, 0, 0, 1'b1); // ramp baseline toward constant input
    send_sample(1000, 0, 0, 1'b1);
    dyn_first = motion_dyn;
    repeat (9) send_sample(1000, 0, 0, 1'b1);
    if (motion_dyn >= dyn_first) $fatal(1, "baseline removal: dyn did not decrease (peak=%0d now=%0d)", dyn_first, motion_dyn);
    if (burst_pulses_seen != 0) $fatal(1, "baseline removal: unexpected bursts=%0d", burst_pulses_seen);

    $display("PASS");
    $finish;
  end

  initial begin
    $dumpfile("motion_preprocess_tb.vcd");
    $dumpvars(0, motion_preprocess_tb);
  end

endmodule
