`timescale 1ns/1ps

module test_mmio #(
    parameter BASE_ADDR = 32'h0300_F000
)(
    input  wire        clk,
    input  wire        resetn,

    input  wire        mem_valid,
    input  wire [31:0] mem_addr,
    input  wire [31:0] mem_wdata,
    input  wire [3:0]  mem_wstrb,

    output reg         mem_ready,
    output reg  [31:0] mem_rdata,

    output reg  [31:0] status_o,
    output reg  [31:0] code_o
);
    wire sel = mem_valid && (mem_addr[31:12] == BASE_ADDR[31:12]);
    wire [31:0] off = mem_addr - BASE_ADDR;

    localparam OFF_STATUS = 32'h0;
    localparam OFF_CODE   = 32'h4;

    always @(posedge clk) begin
        if (!resetn) begin
            mem_ready <= 1'b0;
            mem_rdata <= 32'h0;
            status_o  <= 32'h0;
            code_o    <= 32'h0;
        end else begin
            mem_ready <= 1'b0;

            if (sel) begin
                mem_ready <= 1'b1;

                case (off)
                    OFF_STATUS: mem_rdata <= status_o;
                    OFF_CODE:   mem_rdata <= code_o;
                    default:    mem_rdata <= 32'h0;
                endcase

                if (mem_wstrb != 4'b0000) begin
                    case (off)
                        OFF_STATUS: status_o <= mem_wdata;
                        OFF_CODE:   code_o   <= mem_wdata;
                        default: begin end
                    endcase
                end
            end
        end
    end
endmodule
