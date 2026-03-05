`timescale 1ns/1ps

module cpu_to_ml #(
  parameter ML_BASE = 32'h0300_4000
)(
  input  wire        clk,
  input  wire        resetn,

  // Expose these to cocotb (fake CPU MMIO bus)
  input  wire        mem_valid,
  input  wire [31:0] mem_addr,
  input  wire [31:0] mem_wdata,
  input  wire [3:0]  mem_wstrb,
  output wire        mem_ready,
  output wire [31:0] mem_rdata,

  // Interrupt from ML block (taketwo)
  output wire        ml_irq,


  // Expose taketwo AXI master ("maxi") to cocotb RAM model
  // Write address channel
  output wire [31:0] maxi_awaddr,
  output wire [7:0]  maxi_awlen,
  output wire [2:0]  maxi_awsize,
  output wire [1:0]  maxi_awburst,
  output wire [0:0]  maxi_awlock,
  output wire [3:0]  maxi_awcache,
  output wire [2:0]  maxi_awprot,
  output wire [3:0]  maxi_awqos,
  output wire [1:0]  maxi_awuser,
  output wire        maxi_awvalid,
  input  wire        maxi_awready,

  // Write data channel
  output wire [31:0] maxi_wdata,
  output wire [3:0]  maxi_wstrb,
  output wire        maxi_wlast,
  output wire        maxi_wvalid,
  input  wire        maxi_wready,

  // Write response channel
  input  wire [0:0]  maxi_bid,
  input  wire [1:0]  maxi_bresp,
  input  wire        maxi_bvalid,
  output wire        maxi_bready,

  // Read address channel
  output wire [31:0] maxi_araddr,
  output wire [7:0]  maxi_arlen,
  output wire [2:0]  maxi_arsize,
  output wire [1:0]  maxi_arburst,
  output wire [0:0]  maxi_arlock,
  output wire [3:0]  maxi_arcache,
  output wire [2:0]  maxi_arprot,
  output wire [3:0]  maxi_arqos,
  output wire [1:0]  maxi_aruser,
  output wire        maxi_arvalid,
  input  wire        maxi_arready,

  // Read data channel
  input  wire [0:0]  maxi_rid,
  input  wire [31:0] maxi_rdata,
  input  wire [1:0]  maxi_rresp,
  input  wire        maxi_rlast,
  input  wire        maxi_rvalid,
  output wire        maxi_rready
);


  // AXI-Lite wires between bridge (master) and taketwo_wrap (slave)
  wire [31:0] saxi_awaddr, saxi_wdata, saxi_araddr;
  wire [2:0]  saxi_awprot, saxi_arprot;
  wire        saxi_awvalid, saxi_awready;
  wire [3:0]  saxi_wstrb;
  wire        saxi_wvalid, saxi_wready;
  wire [1:0]  saxi_bresp;
  wire        saxi_bvalid, saxi_bready;
  wire [31:0] saxi_rdata;
  wire [1:0]  saxi_rresp;
  wire        saxi_rvalid, saxi_rready;

  // Optional outputs from bridge (not required for basic AXI-Lite testing)
  wire        ml_event;
  wire [31:0] ml_score;

  ml_axil_bridge_mmio #(.BASE_ADDR(ML_BASE)) u_bridge (
    .clk(clk),
    .resetn(resetn),

    .mem_valid(mem_valid),
    .mem_addr (mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),

    .mem_ready(mem_ready),
    .mem_rdata(mem_rdata),

    .event_o(ml_event),
    .score_o(ml_score),

    .saxi_awaddr (saxi_awaddr),
    .saxi_awprot (saxi_awprot),
    .saxi_awvalid(saxi_awvalid),
    .saxi_awready(saxi_awready),

    .saxi_wdata  (saxi_wdata),
    .saxi_wstrb  (saxi_wstrb),
    .saxi_wvalid (saxi_wvalid),
    .saxi_wready (saxi_wready),

    .saxi_bresp  (saxi_bresp),
    .saxi_bvalid (saxi_bvalid),
    .saxi_bready (saxi_bready),

    .saxi_araddr (saxi_araddr),
    .saxi_arprot (saxi_arprot),
    .saxi_arvalid(saxi_arvalid),
    .saxi_arready(saxi_arready),

    .saxi_rdata  (saxi_rdata),
    .saxi_rresp  (saxi_rresp),
    .saxi_rvalid (saxi_rvalid),
    .saxi_rready (saxi_rready)
  );

  // taketwo wrapper (AXI-Lite slave + AXI master for RAM)
  taketwo_wrap u_taketwo (
    .CLK   (clk),
    .RESETN(resetn),
    .irq   (ml_irq),

    // AXI master (RAM)
    .maxi_awid   (), // wrapper ties IDs to 0 internally
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

    .maxi_arid   (), // wrapper ties IDs to 0 internally
    .maxi_araddr (maxi_araddr),
    .maxi_arlen  (maxi_arlen),
    .maxi_arsize (maxi_arsize),
    .maxi_arburst(maxi_arburst),
    .maxi_arlock (maxi_arlock),
    .maxi_arcache(maxi_arcache),
    .maxi_arprot (maxi_arprot),
    .maxi_arqos  (maxi_arqos),
    .maxi_aruser (maxi_aruser),
    .maxi_arvalid(maxi_arvalid),
    .maxi_arready(maxi_arready),

    .maxi_rid    (maxi_rid),
    .maxi_rdata  (maxi_rdata),
    .maxi_rresp  (maxi_rresp),
    .maxi_rlast  (maxi_rlast),
    .maxi_rvalid (maxi_rvalid),
    .maxi_rready (maxi_rready),

    // AXI-Lite slave (what your bridge drives)
    .saxi_awaddr (saxi_awaddr),
    .saxi_awprot (saxi_awprot),
    .saxi_awvalid(saxi_awvalid),
    .saxi_awready(saxi_awready),

    .saxi_wdata  (saxi_wdata),
    .saxi_wstrb  (saxi_wstrb),
    .saxi_wvalid (saxi_wvalid),
    .saxi_wready (saxi_wready),

    .saxi_bresp  (saxi_bresp),
    .saxi_bvalid (saxi_bvalid),
    .saxi_bready (saxi_bready),

    .saxi_araddr (saxi_araddr),
    .saxi_arprot (saxi_arprot),
    .saxi_arvalid(saxi_arvalid),
    .saxi_arready(saxi_arready),

    .saxi_rdata  (saxi_rdata),
    .saxi_rresp  (saxi_rresp),
    .saxi_rvalid (saxi_rvalid),
    .saxi_rready (saxi_rready)
  );

endmodule