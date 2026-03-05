`timescale 1ns/1ps

// i2c_slave_lis2dw12.sv
//
// Simulation-only I2C slave model for the LIS2DW12 accelerometer.
//
// Connects to i2c_master.sv via the functional transaction interface.
// Reads pre-generated accel_digital.csv and responds to I2C transactions
// as if it were a real LIS2DW12.
//
// Behavior:
//   - STATUS register (0x27): always returns 0x01 (DRDY=1)
//   - DATA registers (0x28-0x2D): returns next sample from CSV
//     packed as 16-bit left-justified (shifted left 2) per LIS2DW12 spec
//
// CSV format: signed integers, one row per sample: ax,ay,az (14-bit values)

module i2c_slave_lis2dw12 #(
    parameter [6:0]  I2C_ADDR  = 7'h18,
    parameter string CSV_FILE  = "sim/data/accel_digital.csv",
    parameter [7:0]  REG_STATUS = 8'h27,
    parameter [7:0]  REG_DATA   = 8'h28
)(
    input  wire        clk,
    input  wire        resetn,

    // Functional I2C bus — connects to i2c_master
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

    // File handle
    int fd;
    int r;

    // Sample storage
    int raw_ax, raw_ay, raw_az;
    reg [15:0] ax16, ay16, az16;   // left-justified 16-bit

    // Response state
    typedef enum logic [1:0] {
        RSP_IDLE    = 2'd0,
        RSP_STATUS  = 2'd1,
        RSP_DATA    = 2'd2
    } rsp_state_t;

    rsp_state_t rsp_state;
    reg [2:0] byte_cnt;

    initial begin
        fd = $fopen(CSV_FILE, "r");
        if (fd == 0) begin
            $display("ERROR: i2c_slave_lis2dw12: cannot open %s", CSV_FILE);
            $fatal(1);
        end
        $display("i2c_slave_lis2dw12: opened %s", CSV_FILE);
    end

    // Read next sample from CSV and pack to 16-bit left-justified
    task read_next_sample;
        r = $fscanf(fd, "%d,%d,%d\n", raw_ax, raw_ay, raw_az);
        if (r == 3) begin
            // Left-justify 14-bit value in 16-bit word (shift left 2)
            ax16 = {raw_ax[13:0], 2'b00};
            ay16 = {raw_ay[13:0], 2'b00};
            az16 = {raw_az[13:0], 2'b00};
        end else begin
            ax16 = 16'h0;
            ay16 = 16'h0;
            az16 = 16'h0;
            $display("i2c_slave_lis2dw12: EOF or read error at time %0t", $time);
        end
    endtask

    always @(posedge clk) begin
        if (!resetn) begin
            i2c_ack    <= 1'b0;
            i2c_rdata  <= 8'h00;
            i2c_rvalid <= 1'b0;
            i2c_rlast  <= 1'b0;
            i2c_err    <= 1'b0;
            rsp_state  <= RSP_IDLE;
            byte_cnt   <= '0;
        end else begin
            i2c_ack    <= 1'b0;
            i2c_rvalid <= 1'b0;
            i2c_rlast  <= 1'b0;
            i2c_err    <= 1'b0;

            case (rsp_state)

                RSP_IDLE: begin
                    if (i2c_req && (i2c_addr == I2C_ADDR)) begin
                        i2c_ack <= 1'b1;
                        if (i2c_reg == REG_STATUS) begin
                            rsp_state <= RSP_STATUS;
                        end else if (i2c_reg == REG_DATA) begin
                            read_next_sample;
                            byte_cnt  <= 3'd0;
                            rsp_state <= RSP_DATA;
                        end else begin
                            // Unknown register — return 0
                            i2c_rdata  <= 8'h00;
                            i2c_rvalid <= 1'b1;
                            i2c_rlast  <= 1'b1;
                        end
                    end
                end

                RSP_STATUS: begin
                    // DRDY always 1 in simulation
                    i2c_rdata  <= 8'h01;
                    i2c_rvalid <= 1'b1;
                    i2c_rlast  <= 1'b1;
                    rsp_state  <= RSP_IDLE;
                end

                RSP_DATA: begin
                    // Send 6 bytes: XL, XH, YL, YH, ZL, ZH
                    i2c_rvalid <= 1'b1;
                    case (byte_cnt)
                        3'd0: i2c_rdata <= ax16[7:0];   // XL
                        3'd1: i2c_rdata <= ax16[15:8];  // XH
                        3'd2: i2c_rdata <= ay16[7:0];   // YL
                        3'd3: i2c_rdata <= ay16[15:8];  // YH
                        3'd4: i2c_rdata <= az16[7:0];   // ZL
                        3'd5: i2c_rdata <= az16[15:8];  // ZH
                        default: i2c_rdata <= 8'h00;
                    endcase

                    if (byte_cnt == 3'd5) begin
                        i2c_rlast <= 1'b1;
                        rsp_state <= RSP_IDLE;
                    end
                    byte_cnt <= byte_cnt + 1;
                end

                default: rsp_state <= RSP_IDLE;
            endcase
        end
    end

endmodule
