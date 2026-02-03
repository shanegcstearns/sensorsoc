`timescale 1ns/1ps

module sim_top_tb;

    reg clk = 0;
    reg resetn = 0;

    wire [7:0] gpio_out;
    wire cpu_clk;
    wire cpu_awake;

    // Change this to your built firmware hex file.
    // Format: one 32-bit hex word per line (like PicoSoC firmware.hex)
    localparam FW = "firmware.hex";

    soc_core #(
        .MEM_WORDS(1024),
        .FIRMWARE_HEX(FW)
    ) dut (
        .clk(clk),
        .resetn(resetn),
        .gpio_out(gpio_out),
        .cpu_clk_o(cpu_clk),
        .cpu_awake_o(cpu_awake)
    );

    // 50 MHz clock (20ns period)
    always #10 clk = ~clk;

    initial begin
        $dumpfile("soc.vcd");
        $dumpvars(0, sim_top_tb);

        // reset
        resetn = 0;
        repeat (10) @(posedge clk);
        resetn = 1;

        // run for some time
        repeat (5_000_000) @(posedge clk);

        $finish;
    end

endmodule
