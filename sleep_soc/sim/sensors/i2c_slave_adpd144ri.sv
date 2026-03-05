`timescale 1ns/1ps

// i2c_slave_adpd144ri.sv
//
// Simulation-only I2C slave model for the ADPD144RI PPG sensor.
//
// Connects to i2c_master.sv via the functional transaction interface.
// Reads pre-generated ppg_digital.csv and responds to I2C transactions
// as if it were a real ADPD144RI.
//
// ADPD144RI register map (relevant subset):
//   0x00 : STATUS register — bits[4:0]=FIFO_COUNT, bit[6]=FIFO_EMPTY
//   0x60 : Time Slot A data (red 660nm) — 2 bytes little-endian
//   0x64 : Time Slot B data (IR  880nm) — 2 bytes little-endian
//
// We model:
//   STATUS (0x00): always returns count=1 (1 sample ready)
//   Slot A (0x60): red channel, loads new sample from CSV
//   Slot B (0x64): IR channel,  uses same sample as Slot A
//
// CSV format: red_counts,ir_counts (14-bit unsigned integers, no header)

module i2c_slave_adpd144ri #(
    parameter [6:0]  I2C_ADDR   = 7'h64,
    parameter string CSV_FILE   = "sim/data/ppg_digital.csv",
    parameter [7:0]  REG_STATUS = 8'h00,
    parameter [7:0]  REG_SLOT_A = 8'h60,
    parameter [7:0]  REG_SLOT_B = 8'h64
)(
    input  wire        clk,
    input  wire        resetn,

    // Functional I2C bus
    input  wire        i2c_req,
    input  wire [6:0]  i2c_addr,
    input  wire [7:0]  i2c_reg,
    input  wire [7:0]  i2c_len,
    output reg         i2c_ack,
    output reg  [7:0]  i2c_rdata,
    output reg         i2c_rvalid,
    output reg         i2c_rlast,
    output reg         i2c_err
);

    int  fd;
    int  r;
    int  raw_red, raw_ir;

    reg [15:0] red16, ir16;

    typedef enum logic [1:0] {
        RSP_IDLE   = 2'd0,
        RSP_STATUS = 2'd1,
        RSP_DATA   = 2'd2
    } rsp_state_t;

    rsp_state_t rsp_state;
    reg [1:0] byte_cnt;
    reg       serving_red;

    initial begin
        fd = $fopen(CSV_FILE, "r");
        if (fd == 0) begin
            $display("ERROR: i2c_slave_adpd144ri: cannot open %s", CSV_FILE);
            $fatal(1);
        end
        $display("i2c_slave_adpd144ri: opened %s", CSV_FILE);
        red16 = 16'h0;
        ir16  = 16'h0;
    end

    task read_next_sample;
        r = $fscanf(fd, "%d,%d\n", raw_red, raw_ir);
        if (r == 2) begin
            red16 = {2'b00, raw_red[13:0]};
            ir16  = {2'b00, raw_ir[13:0]};
        end else begin
            red16 = 16'h0;
            ir16  = 16'h0;
            $display("i2c_slave_adpd144ri: EOF or read error at time %0t", $time);
        end
    endtask

    always @(posedge clk) begin
        if (!resetn) begin
            i2c_ack     <= 1'b0;
            i2c_rdata   <= 8'h00;
            i2c_rvalid  <= 1'b0;
            i2c_rlast   <= 1'b0;
            i2c_err     <= 1'b0;
            rsp_state   <= RSP_IDLE;
            byte_cnt    <= 2'd0;
            serving_red <= 1'b0;
        end else begin
            i2c_ack    <= 1'b0;
            i2c_rvalid <= 1'b0;
            i2c_rlast  <= 1'b0;
            i2c_err    <= 1'b0;

            case (rsp_state)

                RSP_IDLE: begin
                    if (i2c_req && (i2c_addr == I2C_ADDR)) begin
                        i2c_ack <= 1'b1;
                        byte_cnt <= 2'd0;
                        if (i2c_reg == REG_STATUS) begin
                            rsp_state <= RSP_STATUS;
                        end else if (i2c_reg == REG_SLOT_A) begin
                            read_next_sample;
                            serving_red <= 1'b1;
                            rsp_state   <= RSP_DATA;
                        end else if (i2c_reg == REG_SLOT_B) begin
                            serving_red <= 1'b0;
                            rsp_state   <= RSP_DATA;
                        end else begin
                            i2c_rdata  <= 8'h00;
                            i2c_rvalid <= 1'b1;
                            i2c_rlast  <= 1'b1;
                        end
                    end
                end

                RSP_STATUS: begin
                    i2c_rvalid <= 1'b1;
                    if (byte_cnt == 2'd0) begin
                        i2c_rdata <= 8'h00;
                        byte_cnt  <= 2'd1;
                    end else begin
                        i2c_rdata <= 8'h01;   // FIFO count = 1
                        i2c_rlast <= 1'b1;
                        byte_cnt  <= 2'd0;
                        rsp_state <= RSP_IDLE;
                    end
                end

                RSP_DATA: begin
                    i2c_rvalid <= 1'b1;
                    if (byte_cnt == 2'd0) begin
                        i2c_rdata <= serving_red ? red16[7:0] : ir16[7:0];
                        byte_cnt  <= 2'd1;
                    end else begin
                        i2c_rdata <= serving_red ? red16[15:8] : ir16[15:8];
                        i2c_rlast <= 1'b1;
                        byte_cnt  <= 2'd0;
                        rsp_state <= RSP_IDLE;
                    end
                end

                default: rsp_state <= RSP_IDLE;
            endcase
        end
    end

endmodule
