// wrapper for cocotb (cpu_to_ml_tb.py)
`timescale 1ns/1ps

module sim_cpu_to_ml;

  reg clk = 0;
  reg resetn = 0;

  // Fake CPU MMIO signals (driven by cocotb)
  reg        mem_valid;
  reg [31:0] mem_addr;
  reg [31:0] mem_wdata;
  reg [3:0]  mem_wstrb;
  wire       mem_ready;
  wire [31:0] mem_rdata;

  wire       ml_irq;

 
  // Expose taketwo AXI master ("maxi") signals so cocotb can attach AxiRam

  // Write address channel
  wire [31:0] maxi_awaddr;
  wire [7:0]  maxi_awlen;
  wire [2:0]  maxi_awsize;
  wire [1:0]  maxi_awburst;
  wire [0:0]  maxi_awlock;
  wire [3:0]  maxi_awcache;
  wire [2:0]  maxi_awprot;
  wire [3:0]  maxi_awqos;
  wire [1:0]  maxi_awuser;
  wire        maxi_awvalid;
  reg         maxi_awready;

  // Write data channel
  wire [31:0] maxi_wdata;
  wire [3:0]  maxi_wstrb;
  wire        maxi_wlast;
  wire        maxi_wvalid;
  reg         maxi_wready;

  // Write response channel
  reg  [0:0]  maxi_bid;
  reg  [1:0]  maxi_bresp;
  reg         maxi_bvalid;
  wire        maxi_bready;

  // Read address channel
  wire [31:0] maxi_araddr;
  wire [7:0]  maxi_arlen;
  wire [2:0]  maxi_arsize;
  wire [1:0]  maxi_arburst;
  wire [0:0]  maxi_arlock;
  wire [3:0]  maxi_arcache;
  wire [2:0]  maxi_arprot;
  wire [3:0]  maxi_arqos;
  wire [1:0]  maxi_aruser;
  wire        maxi_arvalid;
  reg         maxi_arready;

  // Read data channel
  reg  [0:0]  maxi_rid;
  reg  [31:0] maxi_rdata;
  reg  [1:0]  maxi_rresp;
  reg         maxi_rlast;
  reg         maxi_rvalid;
  wire        maxi_rready;

  // clock
  always #5 clk = ~clk;  // 100MHz

  // DUT
  cpu_to_ml #(.ML_BASE(32'h0300_4000)) dut (
    .clk(clk),
    .resetn(resetn),

    .mem_valid(mem_valid),
    .mem_addr(mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),
    .mem_ready(mem_ready),
    .mem_rdata(mem_rdata),

    .ml_irq(ml_irq),

    // AXI master ("maxi") hookup
    .maxi_awaddr (maxi_awaddr),
    .maxi_awlen  (maxi_awlen),
    .maxi_awsize (maxi_awsize),
    .maxi_awburst(maxi_awburst),
    .maxi_awlock (maxi_awlock),
    .maxi_awcache(maxi_awcache),
    .maxi_awprot (maxi_awprot),
    .maxi_awqos  (maxi_awqos),
    .maxi_awuser (maxi_awuser),
    .maxi_awvalid(maxi_awvalid),
    .maxi_awready(maxi_awready),

    .maxi_wdata  (maxi_wdata),
    .maxi_wstrb  (maxi_wstrb),
    .maxi_wlast  (maxi_wlast),
    .maxi_wvalid (maxi_wvalid),
    .maxi_wready (maxi_wready),

    .maxi_bid    (maxi_bid),
    .maxi_bresp  (maxi_bresp),
    .maxi_bvalid (maxi_bvalid),
    .maxi_bready (maxi_bready),

    .maxi_araddr (maxi_araddr),
    .maxi_arlen  (maxi_arlen),
    .maxi_arsize (maxi_arsize),
    .maxi_arburst(maxi_arburst),
    .maxi_arlock (maxi_arlock),
    .maxi_arcache(maxi_arcache),
    .maxi_arprot (maxi_arprot),
    .maxi_arqos  (maxi_awqos),  // NOTE: qos widths match; harmless if unused
    .maxi_aruser (maxi_aruser),
    .maxi_arvalid(maxi_arvalid),
    .maxi_arready(maxi_arready),

    .maxi_rid    (maxi_rid),
    .maxi_rdata  (maxi_rdata),
    .maxi_rresp  (maxi_rresp),
    .maxi_rlast  (maxi_rlast),
    .maxi_rvalid (maxi_rvalid),
    .maxi_rready (maxi_rready)
  );

  initial begin
    // init signals (cocotb will override too, but this avoids Xs)
    mem_valid = 0;
    mem_addr  = 0;
    mem_wdata = 0;
    mem_wstrb = 0;

    // Default AXI RAM responses = idle (cocotb AxiRam will drive these later)
    maxi_awready = 0;
    maxi_wready  = 0;
    maxi_bid     = 0;
    maxi_bresp   = 0;
    maxi_bvalid  = 0;

    maxi_arready = 0;
    maxi_rid     = 0;
    maxi_rdata   = 0;
    maxi_rresp   = 0;
    maxi_rlast   = 0;
    maxi_rvalid  = 0;

    // reset pulse
    #100;
    resetn = 1;
  end

endmodule