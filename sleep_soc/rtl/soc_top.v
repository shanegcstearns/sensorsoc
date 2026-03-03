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
    parameter IRQC_BASE   = 32'h0300_5000,
    parameter TEST_BASE = 32'h0300_F000
)(
    input  wire        clk,        // always-on clock
    input  wire        resetn,     // active-low reset (always-on)

    output wire [7:0]  gpio_out,   // good for LEDs in FPGA later

    // optional: expose for waveform/debug
    output wire        cpu_clk_o,
    output wire        cpu_awake_o
);

    //FSM - controlled (request) enable
    reg cpu_clk_en;
    //latched enable (updates only when clk is low)
    reg cpu_clk_en_lat;

    always @(negedge clk or negedge resetn) begin
        if (!resetn)
            cpu_clk_en_lat <= 1'b1;     // start awake
        else
            cpu_clk_en_lat <= cpu_clk_en;
    end

    wire cpu_clk = clk & cpu_clk_en_lat;

    assign cpu_clk_o   = cpu_clk;
    assign cpu_awake_o = cpu_clk_en_lat;

    // PicoRV32 bus
    wire        mem_valid;
    wire        mem_instr;
    wire        mem_ready;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [3:0]  mem_wstrb;
    wire [31:0] mem_rdata;
    wire trap;

    // IRQ wiring is provided by SoC IRQ controller.
    wire [31:0] irq;

    //Tweak these later when I add flash/XIP.
    localparam [31:0] STACKADDR      = 4*MEM_WORDS;
    localparam [31:0] PROGADDR_RESET = 32'h0000_0000;
    localparam [31:0] PROGADDR_IRQ   = 32'h0000_0010;

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
        .ENABLE_IRQ(1),
        .ENABLE_IRQ_QREGS(1)
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
        .irq       (irq),
        .trap      (trap) 
    );

    wire bus_valid = mem_valid && cpu_clk_en_lat;
    wire sram_sel = bus_valid && (mem_addr < 4*MEM_WORDS);
    wire mmio_sel = bus_valid && (mem_addr[31:24] == 8'h03);

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

    // IRQ controller: pending/mask/wake filtering + MMIO visibility.
    wire        irqc_ready;
    wire [31:0] irqc_rdata;
    wire        irqc_wake_req;
    wire [31:0] irq_sources = {30'b0, ml_event, timer_event};

    irq_ctrl_mmio #(.BASE_ADDR(IRQC_BASE)) u_irqc (
        .clk      (clk),
        .resetn   (resetn),
        .mem_valid(mmio_sel),
        .mem_addr (mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_ready(irqc_ready),
        .mem_rdata(irqc_rdata),
        .irq_src_i(irq_sources),
        .irq_o    (irq),
        .wake_req_o(irqc_wake_req)
    );

    // Power controller MMIO: sleep request + wake status/reason
    wire        pwr_ready;
    wire [31:0] pwr_rdata;
    wire        sleep_req;

    // Wake sources (always-on)
    wire [31:0] wake_sources;
    assign wake_sources = irq_sources;

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
        .cpu_awake_i(cpu_clk_en_lat)
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
    wire mmio_ready = gpio_ready | pwr_ready | timer_ready | ml_ready | irqc_ready | test_ready;

    wire [31:0] mmio_rdata =
        gpio_ready  ? gpio_rdata  :
        pwr_ready   ? pwr_rdata   :
        timer_ready ? timer_rdata :
        ml_ready    ? ml_rdata    :
        irqc_ready  ? irqc_rdata  :
        test_ready  ? test_rdata  :
        32'h0000_0000;

    // Overall ready/data to CPU
    assign mem_ready = sram_ready | mmio_ready;
    assign mem_rdata = sram_ready ? sram_rdata : mmio_rdata;

// --------------------------------------
// Sleep / wake state machine (always-on)
// --------------------------------------
reg sleeping;
reg cpu_idle_seen;

reg [31:0] wake_sources_d;

always @(posedge clk) begin
  if (!resetn)
    wake_sources_d <= 32'h0;
  else
    wake_sources_d <= wake_sources;
end

wire [31:0] wake_rise  = wake_sources & ~wake_sources_d;
wire        wake_event = |wake_rise;   // 1-cycle pulse


always @(posedge clk) begin
  if (!resetn) begin
    cpu_clk_en     <= 1'b1;  // start awake
    sleeping       <= 1'b0;
    cpu_idle_seen  <= 1'b0;
  end else begin
    // Track whether we've observed the CPU idle at least once while awake
    if (cpu_clk_en) begin
      cpu_idle_seen <= cpu_idle_seen | (~mem_valid);
    end

    if (sleeping) begin
      // Wake has highest priority when sleeping.
      if (irqc_wake_req || wake_event) begin
        cpu_clk_en    <= 1'b1;
        sleeping      <= 1'b0;
        cpu_idle_seen <= 1'b0; // require a fresh idle observation before sleeping again
      end
    end else begin
      // Only sleep when:
      //  - firmware requested it
      //  - we've observed at least one idle cycle (prevents mid-transaction gating)
      //  - no wake event pending
      if (sleep_req && cpu_idle_seen && !(irqc_wake_req || wake_event)) begin
        cpu_clk_en    <= 1'b0;
        sleeping      <= 1'b1;
        cpu_idle_seen <= 1'b0;
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
            $display("SRAM[0]=%08x SRAM[1]=%08x SRAM[2]=%08x SRAM[3]=%08x",
         mem[0], mem[1], mem[2], mem[3]);

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
