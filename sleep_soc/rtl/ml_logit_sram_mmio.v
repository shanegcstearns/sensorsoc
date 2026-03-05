`timescale 1ns/1ps

// SRAM-like MMIO capture buffer for two ML logit readbacks.
// Latches selected AXI-Lite read data into software-visible registers.
// Useful for simulation/debug capture before final SRAM integration.
module ml_logit_sram_mmio #(
 parameter BASE_ADDR = 32'h0300_4000,
 parameter LOGIT0_OFF = 12'h020,
 parameter LOGIT1_OFF = 12'h024
)(
 input wire clk,
 input wire resetn,

 // CPU MMIO side
 input wire mem_valid,
 input wire [31:0] mem_addr,
 input wire [31:0] mem_wdata,
 input wire [3:0] mem_wstrb,
 output reg mem_ready,
 output reg [31:0] mem_rdata,

 // AXI-Lite read tap from taketwo_wrap register map
 input wire [31:0] tap_araddr_i,
 input wire tap_arvalid_i,
 input wire tap_arready_i,
 input wire [31:0] tap_rdata_i,
 input wire tap_rvalid_i,
 input wire tap_rready_i
);
 localparam OFF_CTRL = 32'h00; // bit0 W1C clear valid bits
 localparam OFF_STATUS = 32'h04; // bit0 logit0_valid, bit1 logit1_valid
 localparam OFF_LOGIT0 = 32'h08; // captured 32-bit word for LOGIT0_OFF
 localparam OFF_LOGIT1 = 32'h0C; // captured 32-bit word for LOGIT1_OFF
 localparam OFF_WCOUNT = 32'h10; // number of captures performed

 wire sel = mem_valid && (mem_addr[31:12] == BASE_ADDR[31:12]);
 wire [31:0] off = mem_addr - BASE_ADDR;

 reg [31:0] sram [0:1];
 reg [1:0] valid_bits;
 reg [31:0] write_count;

 reg [31:0] pending_araddr;
 reg pending_araddr_valid;

 wire tap_ar_fire = tap_arvalid_i && tap_arready_i;
 wire tap_r_fire = tap_rvalid_i && tap_rready_i;

 always @(posedge clk) begin
 if (!resetn) begin
 sram[0] <= 32'h0;
 sram[1] <= 32'h0;
 valid_bits <= 2'b00;
 write_count <= 32'h0;
 pending_araddr <= 32'h0;
 pending_araddr_valid <= 1'b0;
 end else begin
 if (tap_ar_fire) begin
 pending_araddr <= tap_araddr_i;
 pending_araddr_valid <= 1'b1;
 end

 if (tap_r_fire && pending_araddr_valid) begin
 if (pending_araddr[11:0] == LOGIT0_OFF) begin
 sram[0] <= tap_rdata_i;
 valid_bits[0] <= 1'b1;
 write_count <= write_count + 32'd1;
 end
 if (pending_araddr[11:0] == LOGIT1_OFF) begin
 sram[1] <= tap_rdata_i;
 valid_bits[1] <= 1'b1;
 write_count <= write_count + 32'd1;
 end
 pending_araddr_valid <= 1'b0;
 end

 if (sel && (mem_wstrb != 4'b0000) && (off == OFF_CTRL) && mem_wdata[0]) begin
 valid_bits <= 2'b00;
 end
 end
 end

 always @(posedge clk) begin
 if (!resetn) begin
 mem_ready <= 1'b0;
 mem_rdata <= 32'h0;
 end else begin
 mem_ready <= 1'b0;
 if (sel) begin
 mem_ready <= 1'b1;
 case (off)
 OFF_CTRL: mem_rdata <= 32'h0;
 OFF_STATUS: mem_rdata <= {30'b0, valid_bits};
 OFF_LOGIT0: mem_rdata <= sram[0];
 OFF_LOGIT1: mem_rdata <= sram[1];
 OFF_WCOUNT: mem_rdata <= write_count;
 default: mem_rdata <= 32'h0;
 endcase
 end
 end
 end
endmodule
