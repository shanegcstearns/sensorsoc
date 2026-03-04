`timescale 1ns/1ps
module host_i2c_bridge_regs (
    input  wire       clk,
    input  wire       resetn,
    input  wire       wr_en_i,
    input  wire [7:0] wr_addr_i,
    input  wire [7:0] wr_data_i,
    input  wire [7:0] rd_addr_i,
    output reg  [7:0] rd_data_o,
    input  wire       proto_err_i,
    output reg        event_o
);

localparam [7:0] REG_WHOAMI      = 8'h00;
localparam [7:0] REG_VERSION     = 8'h01;
localparam [7:0] REG_STATUS      = 8'h02;
localparam [7:0] REG_CTRL        = 8'h03;
localparam [7:0] REG_IRQ_KICK    = 8'h04;
localparam [7:0] REG_IRQ_COUNT_L = 8'h05;
localparam [7:0] REG_IRQ_COUNT_H = 8'h06;

// STATUS bits
localparam integer ST_IRQ_KICK_SEEN = 0;
localparam integer ST_I2C_RX_ERROR  = 1;
localparam integer ST_I2C_PROTO_ERR = 2;

reg        irq_bridge_en;
reg  [7:0] status;
reg [15:0] irq_count;

wire in_unix_reserved = (wr_addr_i >= 8'h10) && (wr_addr_i <= 8'h17);

always @(posedge clk) begin
    if (!resetn) begin
        irq_bridge_en <= 1'b1;
        status        <= 8'h00;
        irq_count     <= 16'h0000;
        event_o       <= 1'b0;
    end else begin
        event_o <= 1'b0;

        if (proto_err_i) status[ST_I2C_PROTO_ERR] <= 1'b1;

        if (wr_en_i) begin
            case (wr_addr_i)
                REG_STATUS: begin
                    status <= status & ~wr_data_i;  // W1C
                end
                REG_CTRL: begin
                    irq_bridge_en <= wr_data_i[0];
                end
                REG_IRQ_KICK: begin
                    if (wr_data_i[0] && irq_bridge_en) begin
                        event_o               <= 1'b1;
                        status[ST_IRQ_KICK_SEEN] <= 1'b1;
                        irq_count             <= irq_count + 16'd1;
                    end
                end
                default: begin
                    if (!in_unix_reserved) status[ST_I2C_RX_ERROR] <= 1'b1;
                end
            endcase
        end
    end
end

always @* begin
    case (rd_addr_i)
        REG_WHOAMI:      rd_data_o = 8'hA5;
        REG_VERSION:     rd_data_o = 8'h01;
        REG_STATUS:      rd_data_o = status;
        REG_CTRL:        rd_data_o = {7'b0, irq_bridge_en};
        REG_IRQ_KICK:    rd_data_o = 8'h00;
        REG_IRQ_COUNT_L: rd_data_o = irq_count[7:0];
        REG_IRQ_COUNT_H: rd_data_o = irq_count[15:8];
        8'h10,8'h11,8'h12,8'h13,8'h14,8'h15,8'h16,8'h17: rd_data_o = 8'h00;
        default:         rd_data_o = 8'h00;
    endcase
end

endmodule
