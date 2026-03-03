`timescale 1ns/1ps

// i2c_master.sv
//
// Functional I2C master for wristwatch sleep tracker SoC.
//
// Polls two sensors on a shared I2C bus:
//   - LIS2DW12  accel  @ 7-bit addr 0x18  (SA0=0)
//   - ADPD144RI PPG    @ 7-bit addr 0x64
//
// Functional model — abstracts away bit-banging, models correct:
//   - Device addressing, register addressing, multi-byte burst reads
//   - ACK/NACK handling
//
// Outputs clean valid/data streams to downstream blocks:
//   - accel_valid, accel_ax/ay/az  → Accel Reader / motion_preprocess
//   - ppg_valid, ppg_red, ppg_ir   → PPG read + timestamp block

module i2c_master #(
    parameter [6:0] ACCEL_ADDR       = 7'h18,
    parameter [6:0] PPG_ADDR         = 7'h64,

    parameter [7:0] ACCEL_REG_STATUS = 8'h27,
    parameter [7:0] ACCEL_REG_DATA   = 8'h28,

    parameter [7:0] PPG_REG_STATUS   = 8'h00,
    parameter [7:0] PPG_REG_SLOT_A   = 8'h60,
    parameter [7:0] PPG_REG_SLOT_B   = 8'h64,

    // Poll periods in sys clk cycles
    // 25 Hz accel:  50_000_000 / 25  = 2_000_000 cycles
    // 100 Hz PPG:   50_000_000 / 100 =   500_000 cycles
    parameter integer ACCEL_POLL_CYCLES = 2_000_000,
    parameter integer PPG_POLL_CYCLES   =   500_000
)(
    input  wire        clk,
    input  wire        resetn,

    // Functional I2C bus
    output reg         i2c_req,
    output reg  [6:0]  i2c_addr,
    output reg  [7:0]  i2c_reg,
    output reg  [7:0]  i2c_len,
    input  wire        i2c_ack,
    input  wire [7:0]  i2c_rdata,
    input  wire        i2c_rvalid,
    input  wire        i2c_rlast,
    input  wire        i2c_err,

    // Accel output
    output reg         accel_valid,
    output reg signed [13:0] accel_ax,
    output reg signed [13:0] accel_ay,
    output reg signed [13:0] accel_az,

    // PPG output
    output reg         ppg_valid,
    output reg [13:0]  ppg_red,
    output reg [13:0]  ppg_ir,

    // Status
    output reg         accel_err_o,
    output reg         ppg_err_o,

    // Config
    input  wire [1:0]  enable_i
);


    // State machine

    typedef enum logic [3:0] {
        ST_IDLE         = 4'd0,
        ST_ACCEL_STATUS = 4'd1,
        ST_ACCEL_WAIT   = 4'd2,
        ST_ACCEL_DATA   = 4'd3,
        ST_ACCEL_RECV   = 4'd4,
        ST_PPG_STATUS   = 4'd5,
        ST_PPG_WAIT     = 4'd6,
        ST_PPG_SLOT_A   = 4'd7,
        ST_PPG_SLOT_A_R = 4'd8,
        ST_PPG_SLOT_B   = 4'd9,
        ST_PPG_SLOT_B_R = 4'd10
    } state_t;

    state_t state_r;

    // Poll counters — count sys clk cycles directly
    reg [31:0] accel_poll_cnt;
    reg [31:0] ppg_poll_cnt;

    // Byte assembly
    reg [7:0]  accel_bytes [0:4];   // XL,XH,YL,YH,ZL (ZH comes in last rvalid)
    reg [2:0]  accel_byte_idx;
    reg [7:0]  ppg_red_lo;          // store red low byte


    // Poll counters

    always @(posedge clk) begin
        if (!resetn) begin
            accel_poll_cnt <= 32'd0;
            ppg_poll_cnt   <= 32'd0;
        end else begin
            // Accel counter — reset when we start a poll
            if (state_r == ST_ACCEL_STATUS)
                accel_poll_cnt <= 32'd0;
            else if (enable_i[0] && accel_poll_cnt < ACCEL_POLL_CYCLES)
                accel_poll_cnt <= accel_poll_cnt + 1;

            // PPG counter — reset when we start a poll
            if (state_r == ST_PPG_STATUS)
                ppg_poll_cnt <= 32'd0;
            else if (enable_i[1] && ppg_poll_cnt < PPG_POLL_CYCLES)
                ppg_poll_cnt <= ppg_poll_cnt + 1;
        end
    end

    wire accel_poll_due = enable_i[0] && (accel_poll_cnt >= ACCEL_POLL_CYCLES);
    wire ppg_poll_due   = enable_i[1] && (ppg_poll_cnt   >= PPG_POLL_CYCLES);


    // Main state machine

    always @(posedge clk) begin
        if (!resetn) begin
            state_r        <= ST_IDLE;
            i2c_req        <= 1'b0;
            i2c_addr       <= 7'h00;
            i2c_reg        <= 8'h00;
            i2c_len        <= 8'd0;
            accel_valid    <= 1'b0;
            accel_ax       <= '0;
            accel_ay       <= '0;
            accel_az       <= '0;
            ppg_valid      <= 1'b0;
            ppg_red        <= '0;
            ppg_ir         <= '0;
            accel_err_o    <= 1'b0;
            ppg_err_o      <= 1'b0;
            accel_byte_idx <= '0;
            ppg_red_lo     <= 8'h0;
        end else begin
            i2c_req     <= 1'b0;
            accel_valid <= 1'b0;
            ppg_valid   <= 1'b0;

            case (state_r)

                
                ST_IDLE: begin
                    if (accel_poll_due) begin
                        state_r <= ST_ACCEL_STATUS;
                    end else if (ppg_poll_due) begin
                        state_r <= ST_PPG_STATUS;
                    end
                end

                
                // ACCEL: read status register (1 byte)
                
                ST_ACCEL_STATUS: begin
                    i2c_req  <= 1'b1;
                    i2c_addr <= ACCEL_ADDR;
                    i2c_reg  <= ACCEL_REG_STATUS;
                    i2c_len  <= 8'd1;
                    if (i2c_ack)
                        state_r <= ST_ACCEL_WAIT;
                end

                ST_ACCEL_WAIT: begin
                    if (i2c_err) begin
                        accel_err_o <= 1'b1;
                        state_r     <= ST_IDLE;
                    end else if (i2c_rvalid && i2c_rlast) begin
                        if (i2c_rdata[0])
                            state_r <= ST_ACCEL_DATA;
                        else
                            state_r <= ST_IDLE;
                    end
                end

                
                // ACCEL: burst read 6 bytes from OUT_X_L
                
                ST_ACCEL_DATA: begin
                    i2c_req        <= 1'b1;
                    i2c_addr       <= ACCEL_ADDR;
                    i2c_reg        <= ACCEL_REG_DATA;
                    i2c_len        <= 8'd6;
                    accel_byte_idx <= 3'd0;
                    if (i2c_ack)
                        state_r <= ST_ACCEL_RECV;
                end

                ST_ACCEL_RECV: begin
                    if (i2c_err) begin
                        accel_err_o <= 1'b1;
                        state_r     <= ST_IDLE;
                    end else if (i2c_rvalid) begin
                        if (accel_byte_idx < 3'd5)
                            accel_bytes[accel_byte_idx] <= i2c_rdata;

                        accel_byte_idx <= accel_byte_idx + 1;

                        if (i2c_rlast) begin
                            // {XH,XL} >> 2 = 14-bit signed
                            accel_ax    <= $signed({accel_bytes[1], accel_bytes[0]}) >>> 2;
                            accel_ay    <= $signed({accel_bytes[3], accel_bytes[2]}) >>> 2;
                            accel_az    <= $signed({i2c_rdata,      accel_bytes[4]}) >>> 2;
                            accel_valid <= 1'b1;
                            state_r     <= ST_IDLE;
                        end
                    end
                end

                
                // PPG: read status (2 bytes)
                
                ST_PPG_STATUS: begin
                    i2c_req  <= 1'b1;
                    i2c_addr <= PPG_ADDR;
                    i2c_reg  <= PPG_REG_STATUS;
                    i2c_len  <= 8'd2;
                    if (i2c_ack)
                        state_r <= ST_PPG_WAIT;
                end

                ST_PPG_WAIT: begin
                    if (i2c_err) begin
                        ppg_err_o <= 1'b1;
                        state_r   <= ST_IDLE;
                    end else if (i2c_rvalid && i2c_rlast) begin
                        // i2c_rdata is last byte = FIFO count
                        if (i2c_rdata != 8'h00)
                            state_r <= ST_PPG_SLOT_A;
                        else
                            state_r <= ST_IDLE;
                    end
                end

                
                // PPG: read Time Slot A (red, 2 bytes)
                
                ST_PPG_SLOT_A: begin
                    i2c_req  <= 1'b1;
                    i2c_addr <= PPG_ADDR;
                    i2c_reg  <= PPG_REG_SLOT_A;
                    i2c_len  <= 8'd2;
                    if (i2c_ack)
                        state_r <= ST_PPG_SLOT_A_R;
                end

                ST_PPG_SLOT_A_R: begin
                    if (i2c_err) begin
                        ppg_err_o <= 1'b1;
                        state_r   <= ST_IDLE;
                    end else if (i2c_rvalid) begin
                        if (!i2c_rlast) begin
                            ppg_red_lo <= i2c_rdata;   // low byte first
                        end else begin
                            ppg_red  <= {i2c_rdata[5:0], ppg_red_lo};  // 14-bit
                            state_r  <= ST_PPG_SLOT_B;
                        end
                    end
                end

                
                // PPG: read Time Slot B (IR, 2 bytes)
                
                ST_PPG_SLOT_B: begin
                    i2c_req  <= 1'b1;
                    i2c_addr <= PPG_ADDR;
                    i2c_reg  <= PPG_REG_SLOT_B;
                    i2c_len  <= 8'd2;
                    if (i2c_ack)
                        state_r <= ST_PPG_SLOT_B_R;
                end

                ST_PPG_SLOT_B_R: begin
                    if (i2c_err) begin
                        ppg_err_o <= 1'b1;
                        state_r   <= ST_IDLE;
                    end else if (i2c_rvalid) begin
                        if (!i2c_rlast) begin
                            ppg_red_lo <= i2c_rdata;   // reuse as ir_lo
                        end else begin
                            ppg_ir    <= {i2c_rdata[5:0], ppg_red_lo};
                            ppg_valid <= 1'b1;
                            state_r   <= ST_IDLE;
                        end
                    end
                end

                default: state_r <= ST_IDLE;
            endcase
        end
    end

endmodule