`timescale 1ns/1ps

module timer_mmio #(
    parameter BASE_ADDR = 32'h0300_2000
)(
    input  wire        clk,
    input  wire        resetn,

    input  wire        mem_valid,
    input  wire [31:0] mem_addr,
    input  wire [31:0] mem_wdata,
    input  wire [3:0]  mem_wstrb,

    output reg         mem_ready,
    output reg  [31:0] mem_rdata,

    output wire        event_o,
    output reg  [31:0] rdata_o
);

    localparam OFF_CTRL   = 32'h0;  // bit0 enable, bit1 periodic
    localparam OFF_RELOAD = 32'h4;
    localparam OFF_COUNT  = 32'h8;
    localparam OFF_EVENT  = 32'hC;  // W1C bit0

    wire sel = mem_valid && (mem_addr[31:12] == BASE_ADDR[31:12]);
    wire [31:0] off = mem_addr - BASE_ADDR;

    reg enable, periodic;
    reg [31:0] reload, count;
    reg event_latched;

    assign event_o = event_latched;

    always @(posedge clk) begin
        if (!resetn) begin
            enable <= 1'b0;
            periodic <= 1'b1;
            reload <= 32'd5_000_000;
            count <= 32'd5_000_000;
            event_latched <= 1'b0;
        end else begin
            if (enable) begin
                if (count != 0) begin
                    count <= count - 1;
                end else begin
                    event_latched <= 1'b1;
                    if (periodic) count <= reload;
                    else enable <= 1'b0;
                end
            end
        end
    end

    // MMIO
    always @(posedge clk) begin
        if (!resetn) begin
            mem_ready <= 1'b0;
            mem_rdata <= 32'h0;
            rdata_o   <= 32'h0;
        end else begin
            mem_ready <= 1'b0;

            if (sel) begin
                mem_ready <= 1'b1;

                case (off)
                    OFF_CTRL:   rdata_o <= {30'b0, periodic, enable};
                    OFF_RELOAD: rdata_o <= reload;
                    OFF_COUNT:  rdata_o <= count;
                    OFF_EVENT:  rdata_o <= {31'b0, event_latched};
                    default:    rdata_o <= 32'h0;
                endcase

                mem_rdata <= rdata_o;

                if (mem_wstrb != 4'b0000) begin
                    case (off)
                        OFF_CTRL: begin
                            if (mem_wstrb[0]) begin
                                enable   <= mem_wdata[0];
                                periodic <= mem_wdata[1];
                            end
                        end
                        OFF_RELOAD: begin
                            reload <= mem_wdata;
                        end
                        OFF_COUNT: begin
                            count <= mem_wdata;
                        end
                        OFF_EVENT: begin
                            // W1C bit0
                            if (mem_wdata[0]) event_latched <= 1'b0;
                        end
                        default: begin end
                    endcase
                end
            end
        end
    end

endmodule
