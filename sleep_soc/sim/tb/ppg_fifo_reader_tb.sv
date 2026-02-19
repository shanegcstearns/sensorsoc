`timescale 1ns/1ps

module ppg_fifo_reader_tb;

  localparam int SAMPLE_W = 16;
  localparam int STATUS_BYTES = 2;
  localparam int COUNT_W = 8;
  localparam int WATERMARK = 4;
  localparam int MAX_BURST = 8;

  logic clk = 0;
  logic resetn = 0;

  logic [31:0] t_now;

  logic        i2c_cmd_valid;
  logic        i2c_cmd_ready;
  logic [6:0]  i2c_cmd_addr;
  logic [7:0]  i2c_cmd_reg;
  logic [7:0]  i2c_cmd_len;

  logic        i2c_rsp_valid;
  logic [7:0]  i2c_rsp_data;
  logic        i2c_rsp_last;
  logic        i2c_rsp_err;
  logic        i2c_rsp_ready;

  wire [SAMPLE_W-1:0] ppg_sample;
  wire                ppg_sample_valid;
  wire [31:0]         ppg_sample_time;

  wire fifo_overflow_flag;
  wire fifo_empty_flag;
  wire i2c_error_flag;

  ppg_fifo_reader #(
    .I2C_ADDR(7'h64),
    .REG_STATUS(8'h06),
    .REG_DATA(8'h08),
    .SAMPLE_W(SAMPLE_W),
    .STATUS_BYTES(STATUS_BYTES),
    .COUNT_W(COUNT_W),
    .WATERMARK(WATERMARK),
    .MAX_BURST_SAMPLES(MAX_BURST),
    .POLL_PERIOD(20),
    .TIMESTAMP_PER_SAMPLE(1)
  ) dut (
    .clk(clk),
    .resetn(resetn),

    .t_now(t_now),

    .i2c_cmd_valid(i2c_cmd_valid),
    .i2c_cmd_ready(i2c_cmd_ready),
    .i2c_cmd_addr(i2c_cmd_addr),
    .i2c_cmd_reg(i2c_cmd_reg),
    .i2c_cmd_len(i2c_cmd_len),

    .i2c_rsp_valid(i2c_rsp_valid),
    .i2c_rsp_data(i2c_rsp_data),
    .i2c_rsp_last(i2c_rsp_last),
    .i2c_rsp_err(i2c_rsp_err),
    .i2c_rsp_ready(i2c_rsp_ready),

    .ppg_sample(ppg_sample),
    .ppg_sample_valid(ppg_sample_valid),
    .ppg_sample_time(ppg_sample_time),

    .fifo_overflow_flag(fifo_overflow_flag),
    .fifo_empty_flag(fifo_empty_flag),
    .i2c_error_flag(i2c_error_flag)
  );

  // 50 MHz
  always #10 clk = ~clk;

  always @(posedge clk) begin
    if (!resetn) t_now <= 32'd0;
    else t_now <= t_now + 1;
  end

  // simple i2c response queue
  byte rsp_q[$];
  bit  rsp_err_q[$];
  bit  rsp_last_q[$];

  task automatic q_push(input byte b, input bit err, input bit last);
    rsp_q.push_back(b);
    rsp_err_q.push_back(err);
    rsp_last_q.push_back(last);
  endtask

  task automatic q_clear();
    rsp_q.delete();
    rsp_err_q.delete();
    rsp_last_q.delete();
  endtask

  assign i2c_cmd_ready = 1'b1;

  // drive response when dut ready and queue not empty
  always @(posedge clk) begin
    i2c_rsp_valid <= 1'b0;
    if (i2c_rsp_ready && (rsp_q.size() != 0)) begin
      i2c_rsp_valid <= 1'b1;
      i2c_rsp_data <= rsp_q.pop_front();
      i2c_rsp_err  <= rsp_err_q.pop_front();
      i2c_rsp_last <= rsp_last_q.pop_front();
    end
  end

  int sample_count;
  logic [SAMPLE_W-1:0] exp_samples[$];
  int                  exp_sample_times[$];
  logic [SAMPLE_W-1:0] got_samples[$];
  int                  got_sample_times[$];

  always @(posedge clk) begin
    if (!resetn) sample_count <= 0;
    else if (ppg_sample_valid) sample_count <= sample_count + 1;
  end

  always @(posedge clk) begin
    if (!resetn) begin
      got_samples.delete();
      got_sample_times.delete();
    end else if (ppg_sample_valid) begin
      got_samples.push_back(ppg_sample);
      got_sample_times.push_back(ppg_sample_time);
      if (!((ppg_sample_time == t_now) || (ppg_sample_time == (t_now - 1)))) begin
        $fatal(1, "timestamp mismatch: got=%0d exp=%0d or %0d", ppg_sample_time, t_now, (t_now - 1));
      end
    end
  end

  task automatic wait_polls(input int n);
    repeat (n) @(posedge clk);
  endtask

  task automatic wait_cmd(input byte reg_addr);
    begin
      while (!(i2c_cmd_valid && i2c_cmd_ready && (i2c_cmd_reg == reg_addr))) begin
        @(posedge clk);
      end
    end
  endtask

  task automatic wait_cmd_check(input byte reg_addr, input byte exp_len, input int timeout);
    int cycles;
    begin
      cycles = 0;
      while (!(i2c_cmd_valid && i2c_cmd_ready && (i2c_cmd_reg == reg_addr))) begin
        @(posedge clk);
        cycles++;
        if (cycles > timeout) $fatal(1, "timeout waiting for cmd reg=0x%02x", reg_addr);
      end
      if (i2c_cmd_addr !== 7'h64) $fatal(1, "cmd addr mismatch: got=0x%02x", i2c_cmd_addr);
      if (i2c_cmd_len !== exp_len) $fatal(1, "cmd len mismatch reg=0x%02x got=%0d exp=%0d", reg_addr, i2c_cmd_len, exp_len);
    end
  endtask

  task automatic assert_no_cmd(input byte reg_addr, input int cycles);
    begin
      repeat (cycles) begin
        @(posedge clk);
        if (i2c_cmd_valid && i2c_cmd_ready && (i2c_cmd_reg == reg_addr)) begin
          $fatal(1, "unexpected cmd reg=0x%02x", reg_addr);
        end
      end
    end
  endtask

  task automatic expect_samples(input int n, input int timeout);
    int cycles;
    begin
      cycles = 0;
      while (got_samples.size() < n) begin
        @(posedge clk);
        cycles++;
        if (cycles > timeout) $fatal(1, "timeout waiting for %0d samples (got %0d)", n, got_samples.size());
      end
    end
  endtask

  task automatic clear_expected();
    exp_samples.delete();
    exp_sample_times.delete();
    got_samples.delete();
    got_sample_times.delete();
  endtask

  task automatic check_expected();
    if (got_samples.size() != exp_samples.size()) begin
      $fatal(1, "sample count mismatch got=%0d exp=%0d", got_samples.size(), exp_samples.size());
    end
    for (int i = 0; i < exp_samples.size(); i++) begin
      if (got_samples[i] !== exp_samples[i]) begin
        $fatal(1, "sample[%0d] mismatch got=0x%0x exp=0x%0x", i, got_samples[i], exp_samples[i]);
      end
    end
  endtask

  initial begin
    i2c_rsp_valid = 0;
    i2c_rsp_data = 0;
    i2c_rsp_err = 0;
    i2c_rsp_last = 0;

    resetn = 0;
    wait_polls(5);
    resetn = 1;

    // 1) fifo empty: status count = 0
    clear_expected();
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    // status bytes: [overflow|count]
    q_push(8'h00, 1'b0, 1'b0);
    q_push(8'h00, 1'b0, 1'b1);
    wait_polls(10);
    if (!fifo_empty_flag) $fatal(1, "empty: fifo_empty_flag not set");
    if (sample_count != 0) $fatal(1, "empty: unexpected samples %0d", sample_count);
    assert_no_cmd(8'h08, 10);

    // 2) below watermark: count=3, no data read
    clear_expected();
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    q_push(8'h03, 1'b0, 1'b0);
    q_push(8'h00, 1'b0, 1'b1);
    wait_polls(10);
    if (fifo_empty_flag) $fatal(1, "below watermark: fifo_empty_flag should be clear");
    assert_no_cmd(8'h08, 10);

    // 3) watermark hit: count=4, send 4 samples
    clear_expected();
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    q_push(8'h04, 1'b0, 1'b0); // count
    q_push(8'h00, 1'b0, 1'b1); // overflow=0
    wait_cmd_check(8'h08, 8, 200);
    q_clear();
    // data bytes: 0x0011, 0x2233, 0x4455, 0x6677
    q_push(8'h11, 1'b0, 1'b0);
    q_push(8'h00, 1'b0, 1'b0);
    q_push(8'h33, 1'b0, 1'b0);
    q_push(8'h22, 1'b0, 1'b0);
    q_push(8'h55, 1'b0, 1'b0);
    q_push(8'h44, 1'b0, 1'b0);
    q_push(8'h77, 1'b0, 1'b0);
    q_push(8'h66, 1'b0, 1'b1);
    exp_samples.push_back(16'h0011);
    exp_samples.push_back(16'h2233);
    exp_samples.push_back(16'h4455);
    exp_samples.push_back(16'h6677);
    expect_samples(4, 200);
    check_expected();

    // 4) max burst limit: count=12, expect 8 samples
    clear_expected();
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    q_push(8'h0C, 1'b0, 1'b0); // count
    q_push(8'h00, 1'b0, 1'b1); // overflow=0
    wait_cmd_check(8'h08, (MAX_BURST*2), 200);
    q_clear();
    for (int i = 0; i < MAX_BURST; i++) begin
      q_push(byte'(i[7:0]), 1'b0, 1'b0);
      q_push(8'h00, 1'b0, (i == (MAX_BURST-1)));
      exp_samples.push_back({8'h00, byte'(i[7:0])});
    end
    expect_samples(MAX_BURST, 400);
    check_expected();

    // 5) i2c error: status read error
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    q_push(8'h00, 1'b1, 1'b1);
    wait_polls(10);
    if (!i2c_error_flag) $fatal(1, "i2c error: flag not set");

    // 6) i2c error: data read error (no samples)
    clear_expected();
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    q_push(8'h04, 1'b0, 1'b0);
    q_push(8'h00, 1'b0, 1'b1);
    wait_cmd_check(8'h08, 8, 200);
    q_clear();
    q_push(8'hAA, 1'b1, 1'b1);
    wait_polls(20);
    if (got_samples.size() != 0) $fatal(1, "data error: unexpected samples");
    if (!i2c_error_flag) $fatal(1, "data error: i2c_error_flag not set");

    // 7) overflow flag set
    wait_cmd_check(8'h06, STATUS_BYTES[7:0], 200);
    q_clear();
    q_push(8'h02, 1'b0, 1'b0); // count
    q_push(8'h80, 1'b0, 1'b1); // overflow bit high (msb)
    wait_polls(10);
    if (!fifo_overflow_flag) $fatal(1, "overflow: flag not set");

    $display("PASS");
    $finish;
  end

  initial begin
    $dumpfile("ppg_fifo_reader_tb.vcd");
    $dumpvars(0, ppg_fifo_reader_tb);
  end

endmodule
