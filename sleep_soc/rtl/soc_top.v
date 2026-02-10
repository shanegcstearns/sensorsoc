`timescale 1ns/1ps

module soc_top #(
    // SRAM: WORDS x 32-bit. Default 4 KB.
    parameter integer MEM_WORDS = 1024,

    // Firmware init file (one 32-bit word per line, hex).
    parameter FIRMWARE_HEX = "",

    // Base addresses (MMIO region)
    parameter GPIO_BASE   = 32'h0300_0000,
    parameter PWR_BASE    = 32'h0300_1000,
    parameter TIMER_BASE  = 32'h0300_2000,
    parameter ML_BASE     = 32'h0300_3000,
    parameter TEST_BASE = 32'h0300_F000
)(
    input  wire        clk,        // always-on clock
    input  wire        resetn,     // active-low reset (always-on)

    output wire [7:0]  gpio_out,   // good for LEDs in FPGA later

    // optional: expose for waveform/debug
    output wire        cpu_clk_o,
    output wire        cpu_awake_o
);

    // CPU clock gating (always-on wake logic controls cpu_clk_en)
    reg cpu_clk_en;

    // simple gated clock for simulation
    wire cpu_clk = clk & cpu_clk_en;
    assign cpu_clk_o   = cpu_clk;
    assign cpu_awake_o = cpu_clk_en;

    // PicoRV32 bus
    wire        mem_valid;
    wire        mem_instr;
    wire        mem_ready;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [3:0]  mem_wstrb;
    wire [31:0] mem_rdata;

    // IRQ not required yet, add later
    wire [31:0] irq = 32'b0;

    //Tweak these later when I add flash/XIP.
    localparam [31:0] STACKADDR      = 4*MEM_WORDS;
    localparam [31:0] PROGADDR_RESET = 32'h0000_0000;
    localparam [31:0] PROGADDR_IRQ   = 32'h0000_0000;

    picorv32 #(
        .STACKADDR(STACKADDR),
        .PROGADDR_RESET(PROGADDR_RESET),
        .PROGADDR_IRQ(PROGADDR_IRQ),
        .BARREL_SHIFTER(1),
        .COMPRESSED_ISA(1),
        .ENABLE_COUNTERS(1),
        .ENABLE_MUL(1),
        .ENABLE_DIV(1),
        .ENABLE_FAST_MUL(0),
        .ENABLE_IRQ(0),
        .ENABLE_IRQ_QREGS(0)
    ) cpu (
        .clk       (cpu_clk),
        .resetn    (resetn),
        .mem_valid (mem_valid),
        .mem_instr (mem_instr),
        .mem_ready (mem_ready),
        .mem_addr  (mem_addr),
        .mem_wdata (mem_wdata),
        .mem_wstrb (mem_wstrb),
        .mem_rdata (mem_rdata),
        .irq       (irq)
    );

    // Address map
    //   SRAM: 0x0000_0000 .. 0x0000_(4*MEM_WORDS-1)
    //   MMIO: 0x0300_0000+
    // SRAM select
    wire sram_sel = mem_valid && (mem_addr < 4*MEM_WORDS);

    // MMIO select
    wire mmio_sel = mem_valid && (mem_addr[31:24] == 8'h03);

    // SRAM
    wire        sram_ready;
    wire [31:0] sram_rdata;

    simple_sram #(
        .WORDS(MEM_WORDS),
        .INIT_HEX(FIRMWARE_HEX)
    ) sram (
        .clk   (cpu_clk),
        .resetn(resetn),
        .valid (sram_sel),
        .ready (sram_ready),
        .wstrb (mem_wstrb),
        .addr  (mem_addr),
        .wdata (mem_wdata),
        .rdata (sram_rdata)
    );

    // MMIO blocks (always-on clock)
    // Put MMIO on always-on clk so it can wake CPU even if gated.
    // But the CPU accesses MMIO via cpu_clk domain signals.
    //
    // In this starter design:
    // - MMIO registers update on always-on clk using mem_valid/etc which
    //   only toggle when cpu_clk is running.
    // - Timer and ML stub run on always-on clk and can wake CPU.

    // GPIO MMIO (single register at GPIO_BASE)
    wire        gpio_ready;
    wire [31:0] gpio_rdata;

    gpio_mmio #(.BASE_ADDR(GPIO_BASE)) u_gpio (
        .clk      (clk),
        .resetn   (resetn),
        .mem_valid(mmio_sel),
        .mem_addr (mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_ready(gpio_ready),
        .mem_rdata(gpio_rdata),
        .gpio_out (gpio_out)
    );

    // Timer wake block
    wire        timer_ready;
    wire [31:0] timer_rdata;
    wire        timer_event;

    timer_mmio #(.BASE_ADDR(TIMER_BASE)) u_timer (
        .clk      (clk),
        .resetn   (resetn),
        .mem_valid(mmio_sel),
        .mem_addr (mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_ready(timer_ready),
        .mem_rdata(),
        .event_o  (timer_event),
        .rdata_o  (timer_rdata)
    );

    // ML stub block
    wire        ml_ready;
    wire [31:0] ml_rdata;
    wire        ml_event;
    wire [31:0] ml_score;

    ml_stub_mmio #(.BASE_ADDR(ML_BASE)) u_ml (
        .clk      (clk),
        .resetn   (resetn),
        .mem_valid(mmio_sel),
        .mem_addr (mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_ready(ml_ready),
        .mem_rdata(ml_rdata),
        .event_o  (ml_event),
        .score_o  (ml_score)
    );

    // Power controller MMIO: sleep request + wake status/reason
    wire        pwr_ready;
    wire [31:0] pwr_rdata;
    wire        sleep_req;

    // Wake sources (always-on)
    wire [31:0] wake_sources;
    assign wake_sources = {30'b0, ml_event, timer_event}; // bit1=ML, bit0=timer

    pwrctrl_mmio #(.BASE_ADDR(PWR_BASE)) u_pwr (
        .clk        (clk),
        .resetn     (resetn),
        .mem_valid  (mmio_sel),
        .mem_addr   (mem_addr),
        .mem_wdata  (mem_wdata),
        .mem_wstrb  (mem_wstrb),
        .mem_ready  (pwr_ready),
        .mem_rdata  (pwr_rdata),
        .sleep_req_o(sleep_req),
        .wake_src_i (wake_sources),
        .cpu_awake_i(cpu_clk_en)
    );

    wire        test_ready;
    wire [31:0] test_rdata;
    wire [31:0] test_status, test_code;

    test_mmio #(.BASE_ADDR(TEST_BASE)) u_test (
        .clk(clk), .resetn(resetn),
        .mem_valid(mmio_sel),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_ready(test_ready),
        .mem_rdata(test_rdata),
        .status_o(test_status),
        .code_o(test_code)
    );




    // MMIO bus response mux (to PicoRV32)
    wire mmio_ready = gpio_ready | pwr_ready | timer_ready | ml_ready | test_ready;

    wire [31:0] mmio_rdata =
        gpio_ready  ? gpio_rdata  :
        pwr_ready   ? pwr_rdata   :
        timer_ready ? timer_rdata :
        ml_ready    ? ml_rdata    :
        test_ready  ? test_rdata  :
        32'h0000_0000;

    // Overall ready/data to CPU
    assign mem_ready = sram_ready | mmio_ready;
    assign mem_rdata = sram_ready ? sram_rdata : mmio_rdata;


    // Sleep / wake state machine (always-on)
    // - Gate CPU clock when firmware sets SLEEP_REQ and CPU is idle
    // - Ungate CPU clock when any wake source asserts
    // Consider "CPU idle" when it is not issuing a mem transaction.
    // Sample this on always-on clock while CPU is awake.
    reg cpu_idle_seen;

    always @(posedge clk) begin
        if (!resetn) begin
            cpu_idle_seen <= 1'b0;
        end else if (cpu_clk_en) begin
            cpu_idle_seen <= (mem_valid == 1'b0);
        end
    end

    wire wake_event = |wake_sources;

    always @(posedge clk) begin
        if (!resetn) begin
            cpu_clk_en <= 1'b1;      // start awake
        end else begin
            // Wake has priority
            if (wake_event) begin
                cpu_clk_en <= 1'b1;
            end else if (cpu_clk_en) begin
                // Only gate if firmware asked + CPU appears idle
                if (sleep_req && cpu_idle_seen) begin
                    cpu_clk_en <= 1'b0;
                end
            end
        end
    end

endmodule


// Simple synchronous SRAM with optional init file for simulation
// - Responds in 1 cycle when valid
// - Supports byte writes via wstrb
module simple_sram #(
    parameter integer WORDS = 1024,
    parameter INIT_HEX = ""
)(
    input  wire        clk,
    input  wire        resetn,

    input  wire        valid,
    output reg         ready,

    input  wire [3:0]  wstrb,
    input  wire [31:0] addr,   // byte address
    input  wire [31:0] wdata,
    output reg  [31:0] rdata
);
    reg [31:0] mem [0:WORDS-1];

    // optional init
    integer i;
    initial begin
        if (INIT_HEX != "") begin
            $display("simple_sram: loading INIT_HEX=%s", INIT_HEX);
            $readmemh(INIT_HEX, mem);
        end else begin
            // default clear (optional)
            for (i = 0; i < WORDS; i = i + 1)
                mem[i] = 32'h0000_0000;
        end
    end

    wire [31:0] word_index = addr >> 2;

    always @(posedge clk) begin
        if (!resetn) begin
            ready <= 1'b0;
            rdata <= 32'h0;
        end else begin
            ready <= 1'b0;
            if (valid) begin
                ready <= 1'b1;
                rdata <= mem[word_index];

                if (wstrb[0]) mem[word_index][ 7: 0] <= wdata[ 7: 0];
                if (wstrb[1]) mem[word_index][15: 8] <= wdata[15: 8];
                if (wstrb[2]) mem[word_index][23:16] <= wdata[23:16];
                if (wstrb[3]) mem[word_index][31:24] <= wdata[31:24];
            end
        end
    end
endmodule
