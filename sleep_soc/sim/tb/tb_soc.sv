`timescale 1ns/1ps

module tb_soc;

  localparam string FW = "firmware/build/firmware.hex";

<<<<<<< HEAD
  // Optional: page constants for runtime filtering (NOT hierarchical)
  localparam logic [31:12] TIMER_PAGE = 20'h03002;
  localparam logic [31:12] PWR_PAGE   = 20'h03001;
  localparam logic [31:12] TEST_PAGE  = 20'h0300F;

=======
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
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

<<<<<<< HEAD
  // --------------------------------------------------
  // Sensor file players
  // --------------------------------------------------

  logic        accel_valid, accel_ok;
  logic signed [13:0] ax, ay, az;

  accel_file_player accel_src (
      .clk(clk),
      .resetn(resetn),
      .sample_valid(accel_valid),
      .sample_ok(accel_ok),
      .ax(ax),
      .ay(ay),
      .az(az)
  );

  logic        ppg_valid;
  logic [13:0] ppg_red;
  logic [13:0] ppg_ir;

  ppg_file_player ppg_src (
      .clk(clk),
      .resetn(resetn),
      .sample_valid(ppg_valid),
      .red_counts(ppg_red),
      .ir_counts(ppg_ir)
  );

  // ------------------------------------------------------------
  // TAP internal DUT signals into TB wires (Icarus-safe pattern)
  // ------------------------------------------------------------
  // NOTE: These are *continuous assigns*, not params/generate.
  wire        tap_trap      = dut.trap;

  wire        tap_mem_valid = dut.mem_valid;
  wire [31:0] tap_mem_addr  = dut.mem_addr;
  wire [31:0] tap_mem_wdata = dut.mem_wdata;
  wire [3:0]  tap_mem_wstrb = dut.mem_wstrb;
  wire        tap_mmio_sel  = dut.mmio_sel;

  wire [31:0] tap_test_status = dut.test_status;
  wire [31:0] tap_test_code   = dut.test_code;

=======
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
  // ----------------------------
  // Sleep/Wake observation flags
  // ----------------------------
  bit saw_sleep = 0;
  bit saw_wake  = 0;

<<<<<<< HEAD
=======
  // Track cpu_awake_o transitions
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);

<<<<<<< HEAD
      if (!cpu_awake_o)
        saw_sleep = 1;

=======
      // sleep seen when cpu_awake_o goes low at least once
      if (!cpu_awake_o)
        saw_sleep = 1;

      // wake seen when it returns high after sleep
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
      if (saw_sleep && cpu_awake_o)
        saw_wake = 1;
    end
  end

<<<<<<< HEAD
  // timeout watchdog
=======
  // timeout watchdog (adjust as needed)
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
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

<<<<<<< HEAD
  // PASS/FAIL monitor
=======
  // PASS/FAIL monitor (requires test_mmio block in DUT)
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
  initial begin
    wait(resetn == 1);

    forever begin
      @(posedge clk);

<<<<<<< HEAD
      if (tap_test_status == 32'hCAFE_BABE) begin
=======
      if (dut.test_status == 32'hCAFE_BABE) begin
        // Require real sleep/wake if this is your sleep-wake firmware test
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
        if (!saw_sleep || !saw_wake) begin
          $display("FAIL: Firmware reported PASS but did not observe cpu_awake_o 1->0->1");
          $display("  saw_sleep=%0d saw_wake=%0d", saw_sleep, saw_wake);
          $fatal(1, "Sleep/Wake not observed");
        end

        $display("PASS (sleep/wake observed)");
        $finish;
      end

<<<<<<< HEAD
      if (tap_test_status == 32'hDEAD_BEEF) begin
        $display("FAIL, code=0x%08x", tap_test_code);
=======
      if (dut.test_status == 32'hDEAD_BEEF) begin
        $display("FAIL, code=0x%08x", dut.test_code);
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
        $fatal(1, "Firmware reported FAIL");
      end
    end
  end

  // Optional wave dump
  initial begin
    $dumpfile("soc.vcd");
    $dumpvars(0, tb_soc);
  end

<<<<<<< HEAD
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

  // MMIO write monitor (runtime compare only; NO generate/const expr)
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);

      if (tap_mmio_sel && tap_mem_valid && (tap_mem_wstrb != 4'b0000)) begin
        $display("MMIO WRITE addr=%08x data=%08x wstrb=%b time=%0t",
                 tap_mem_addr, tap_mem_wdata, tap_mem_wstrb, $time);

        // Example: only print for a specific 4KB page:
        // if (tap_mem_addr[31:12] == TIMER_PAGE) begin
        //   $display("  (timer page write)");
        // end
      end
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
=======


  initial begin
  wait(resetn == 1);
  forever begin
    @(posedge clk);
    if (dut.trap) begin
      $display("TRAP asserted at time %0t", $time);
      $fatal(1, "CPU trapped (likely bad code / bad link address / illegal instruction)");
    end
  end
end


initial begin
  wait(resetn == 1);
  forever begin
    @(posedge clk);
    if (dut.mmio_sel && dut.mem_valid && (dut.mem_wstrb != 0)) begin
      $display("MMIO WRITE addr=%08x data=%08x wstrb=%b time=%0t",
               dut.mem_addr, dut.mem_wdata, dut.mem_wstrb, $time);
    end
  end
end









endmodule
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
