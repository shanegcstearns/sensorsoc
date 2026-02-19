`timescale 1ns/1ps

module tb_soc;

  localparam string FW = "firmware/build/firmware.hex";

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

  // ----------------------------
  // Sleep/Wake observation flags
  // ----------------------------
  bit saw_sleep = 0;
  bit saw_wake  = 0;

  // Track cpu_awake_o transitions
  initial begin
    wait(resetn == 1);
    forever begin
      @(posedge clk);

      // sleep seen when cpu_awake_o goes low at least once
      if (!cpu_awake_o)
        saw_sleep = 1;

      // wake seen when it returns high after sleep
      if (saw_sleep && cpu_awake_o)
        saw_wake = 1;
    end
  end

  // timeout watchdog (adjust as needed)
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

  // PASS/FAIL monitor (requires test_mmio block in DUT)
  initial begin
    wait(resetn == 1);

    forever begin
      @(posedge clk);

      if (dut.test_status == 32'hCAFE_BABE) begin
        // Require real sleep/wake if this is your sleep-wake firmware test
        if (!saw_sleep || !saw_wake) begin
          $display("FAIL: Firmware reported PASS but did not observe cpu_awake_o 1->0->1");
          $display("  saw_sleep=%0d saw_wake=%0d", saw_sleep, saw_wake);
          $fatal(1, "Sleep/Wake not observed");
        end

        $display("PASS (sleep/wake observed)");
        $finish;
      end

      if (dut.test_status == 32'hDEAD_BEEF) begin
        $display("FAIL, code=0x%08x", dut.test_code);
        $fatal(1, "Firmware reported FAIL");
      end
    end
  end

  // Optional wave dump
  initial begin
    $dumpfile("soc.vcd");
    $dumpvars(0, tb_soc);
  end



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
