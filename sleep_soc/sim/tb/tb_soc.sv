`timescale 1ns/1ps

module tb_soc;

  localparam string FW = "firmware/build/firmware.hex";

  // Optional: page constants for runtime filtering (NOT hierarchical)
  localparam logic [31:12] TIMER_PAGE = 20'h03002;
  localparam logic [31:12] PWR_PAGE   = 20'h03001;
  localparam logic [31:12] TEST_PAGE  = 20'h0300F;

  logic clk = 0;
  logic resetn = 0;

  wire [7:0] gpio_out;
  wire cpu_clk_o, cpu_awake_o;

  // DUT
  soc_top #(
    .MEM_WORDS(1024),
    .FIRMWARE_HEX(FW)
  ) dut (
    .clk(clk),
    .resetn(resetn),
    .gpio_out(gpio_out),
    .cpu_clk_o(cpu_clk_o),
    .cpu_awake_o(cpu_awake_o)
  );

  // 50 MHz clock
  always #10 clk = ~clk;

  // reset
  initial begin
    resetn = 0;
    repeat (10) @(posedge clk);
    resetn = 1;
  end

  // --------------------------------------------------
  // Shared functional I2C bus wires
  // --------------------------------------------------
  wire        i2c_req;
  wire [6:0]  i2c_addr;
  wire [7:0]  i2c_reg;
  wire [7:0]  i2c_len;

  // Accel slave response wires
  wire        accel_i2c_ack;
  wire [7:0]  accel_i2c_rdata;
  wire        accel_i2c_rvalid;
  wire        accel_i2c_rlast;
  wire        accel_i2c_err;

  // PPG slave response wires
  wire        ppg_i2c_ack;
  wire [7:0]  ppg_i2c_rdata;
  wire        ppg_i2c_rvalid;
  wire        ppg_i2c_rlast;
  wire        ppg_i2c_err;

  // Mux slave responses to master based on address
  wire        i2c_ack    = (i2c_addr == 7'h18) ? accel_i2c_ack    : ppg_i2c_ack;
  wire [7:0]  i2c_rdata  = (i2c_addr == 7'h18) ? accel_i2c_rdata  : ppg_i2c_rdata;
  wire        i2c_rvalid = (i2c_addr == 7'h18) ? accel_i2c_rvalid : ppg_i2c_rvalid;
  wire        i2c_rlast  = (i2c_addr == 7'h18) ? accel_i2c_rlast  : ppg_i2c_rlast;
  wire        i2c_err    = (i2c_addr == 7'h18) ? accel_i2c_err    : ppg_i2c_err;

  // --------------------------------------------------
  // I2C master
  // --------------------------------------------------
  wire        accel_valid;
  wire signed [13:0] accel_ax, accel_ay, accel_az;
  wire        ppg_valid;
  wire [13:0] ppg_red, ppg_ir;
  wire        accel_err, ppg_err;

  i2c_master u_i2c_master (
      .clk        (clk),
      .resetn     (resetn),
      .i2c_req    (i2c_req),
      .i2c_addr   (i2c_addr),
      .i2c_reg    (i2c_reg),
      .i2c_len    (i2c_len),
      .i2c_ack    (i2c_ack),
      .i2c_rdata  (i2c_rdata),
      .i2c_rvalid (i2c_rvalid),
      .i2c_rlast  (i2c_rlast),
      .i2c_err    (i2c_err),
      .accel_valid(accel_valid),
      .accel_ax   (accel_ax),
      .accel_ay   (accel_ay),
      .accel_az   (accel_az),
      .ppg_valid  (ppg_valid),
      .ppg_red    (ppg_red),
      .ppg_ir     (ppg_ir),
      .accel_err_o(accel_err),
      .ppg_err_o  (ppg_err),
      .enable_i   (2'b11)
  );

  // --------------------------------------------------
  // I2C slave models
  // --------------------------------------------------
  i2c_slave_lis2dw12 u_accel_slave (
      .clk        (clk),
      .resetn     (resetn),
      .i2c_req    (i2c_req),
      .i2c_addr   (i2c_addr),
      .i2c_reg    (i2c_reg),
      .i2c_len    (i2c_len),
      .i2c_ack    (accel_i2c_ack),
      .i2c_rdata  (accel_i2c_rdata),
      .i2c_rvalid (accel_i2c_rvalid),
      .i2c_rlast  (accel_i2c_rlast),
      .i2c_err    (accel_i2c_err)
  );

  i2c_slave_adpd144ri u_ppg_slave (
      .clk        (clk),
      .resetn     (resetn),
      .i2c_req    (i2c_req),
      .i2c_addr   (i2c_addr),
      .i2c_reg    (i2c_reg),
      .i2c_len    (i2c_len),
      .i2c_ack    (ppg_i2c_ack),
      .i2c_rdata  (ppg_i2c_rdata),
      .i2c_rvalid (ppg_i2c_rvalid),
      .i2c_rlast  (ppg_i2c_rlast),
      .i2c_err    (ppg_i2c_err)
  );

  // --------------------------------------------------
  // Motion preprocessor
  // --------------------------------------------------
  wire        motion_valid;
  wire [15:0] motion_mag;
  wire        epoch_done;
  wire [47:0] motion_energy_epoch;

  motion_preprocess u_motion (
      .clk                    (clk),
      .resetn                 (resetn),
      .sample_valid_i         (accel_valid),
      .sample_ok_i            (1'b1),
      .ax_i                   (accel_ax),
      .ay_i                   (accel_ay),
      .az_i                   (accel_az),
      .cfg_th_hi_i            (17'd500),
      .cfg_th_lo_i            (17'd200),
      .cfg_still_th_i         (17'd100),
      .cfg_debounce_n_i       (8'd3),
      .cfg_epoch_len_i        (16'd25),
      .cfg_epoch_external_i   (1'b0),
      .epoch_end_i            (1'b0),
      .cfg_energy_sq_i        (1'b0),
      .motion_valid_o         (motion_valid),
      .motion_mag_o           (motion_mag),
      .burst_pulse_o          (),
      .in_burst_o             (),
      .stillness_flag_o       (),
      .epoch_done_o           (epoch_done),
      .motion_energy_epoch_o  (motion_energy_epoch),
      .motion_delta_epoch_o   (),
      .burst_count_epoch_o    (),
      .stillness_count_epoch_o(),
      .sample_count_epoch_o   ()
  );

  // ------------------------------------------------------------
  // TAP internal DUT signals into TB wires (Icarus-safe pattern)
  // ------------------------------------------------------------
  wire        tap_trap      = dut.trap;

  wire        tap_mem_valid = dut.mem_valid;
  wire [31:0] tap_mem_addr  = dut.mem_addr;
  wire [31:0] tap_mem_wdata = dut.mem_wdata;
  wire [3:0]  tap_mem_wstrb = dut.mem_wstrb;
  wire        tap_mmio_sel  = dut.mmio_sel;

  wire [31:0] tap_test_status = dut.test_status;
  wire [31:0] tap_test_code   = dut.test_code;

  // ----------------------------
  // Sleep/Wake observation flags
  // ----------------------------
  bit saw_sleep = 0;
  bit saw_wake  = 0;

  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);
      if (!cpu_awake_o)
        saw_sleep = 1;
      if (saw_sleep && cpu_awake_o)
        saw_wake = 1;
    end
  end

  // timeout watchdog
  longint unsigned cycles;
  initial begin
    cycles = 0;
    wait(resetn == 1);
    while (cycles < 20_000_000) begin
      @(posedge clk);
      cycles++;
    end
    $fatal(1, "TIMEOUT: test did not finish");
  end

  // PASS/FAIL monitor
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);

      if (tap_test_status == 32'hCAFE_BABE) begin
        if (!saw_sleep || !saw_wake) begin
          $display("FAIL: Firmware reported PASS but did not observe cpu_awake_o 1->0->1");
          $display("  saw_sleep=%0d saw_wake=%0d", saw_sleep, saw_wake);
          $fatal(1, "Sleep/Wake not observed");
        end
        $display("PASS (sleep/wake observed)");
        $finish;
      end

      if (tap_test_status == 32'hDEAD_BEEF) begin
        $display("FAIL, code=0x%08x", tap_test_code);
        $fatal(1, "Firmware reported FAIL");
      end
    end
  end

  // Optional wave dump
  initial begin
    $dumpfile("soc.vcd");
    $dumpvars(0, tb_soc);
  end

  // Trap monitor
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);
      if (tap_trap) begin
        $display("TRAP asserted at time %0t", $time);
        $fatal(1, "CPU trapped (likely bad code / bad link address / illegal instruction)");
      end
    end
  end

  // MMIO write monitor
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);
      if (tap_mmio_sel && tap_mem_valid && (tap_mem_wstrb != 4'b0000)) begin
        $display("MMIO WRITE addr=%08x data=%08x wstrb=%b time=%0t",
                 tap_mem_addr, tap_mem_wdata, tap_mem_wstrb, $time);
      end
    end
  end

  // Motion + PPG data monitor
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);
      if (epoch_done)
        $display("EPOCH DONE: energy=%0d @ %0t", motion_energy_epoch, $time);
      if (ppg_valid)
        $display("PPG SAMPLE: red=%0d ir=%0d @ %0t", ppg_red, ppg_ir, $time);
    end
  end

  // I2C activity monitor (temp, to see why EPOCH DONE and PPG SAMPLE messages are not printing in sim)
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);
      if (i2c_req)
        $display("I2C REQ: addr=0x%02x reg=0x%02x len=%0d @ %0t",
                 i2c_addr, i2c_reg, i2c_len, $time);
      if (accel_valid)
        $display("ACCEL SAMPLE: ax=%0d ay=%0d az=%0d @ %0t",
                 accel_ax, accel_ay, accel_az, $time);
    end
  end

  // Print when firmware writes PASS/FAIL
  logic [31:0] last_status;
  initial begin
    last_status = 32'h0;
    wait(resetn == 1);
    forever begin
      @(posedge clk);
      if (tap_test_status != last_status) begin
        $display("test_status changed: %08x -> %08x @ %0t",
                 last_status, tap_test_status, $time);
        last_status = tap_test_status;
      end
    end
  end

endmodule
