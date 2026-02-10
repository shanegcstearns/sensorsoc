`timescale 1ns/1ps

module tb_soc;

  localparam string FW = "firmware.hex";

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

  // timeout watchdog (adjust as needed)
  initial begin
    longint unsigned cycles = 0;
    wait(resetn == 1);
    while (cycles < 20_000_000) begin
      @(posedge clk);
      cycles++;
      // Optional: if CPU asleep forever, still allow timer to wake it.
    end
    $fatal(1, "TIMEOUT: test did not finish");
  end

  // PASS/FAIL monitor (requires test_mmio block in DUT)
  initial begin
    wait(resetn == 1);

    // Wait until firmware writes a nonzero status
    forever begin
      @(posedge clk);
      if (dut.test_status == 32'hCAFE_BABE) begin
        $display("PASS");
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

endmodule
