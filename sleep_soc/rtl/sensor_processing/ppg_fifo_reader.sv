`timescale 1ns/1ps

module ppg_fifo_reader #(
    parameter [6:0]  I2C_ADDR = 7'h64,
    parameter [7:0]  REG_STATUS = 8'h00,
    parameter [7:0]  REG_FIFO_THRESH = 8'h06,
    parameter [7:0]  REG_FIFO_ACCESS_ENA = 8'h5F,
    parameter [7:0]  REG_FIFO_ACCESS = 8'h60,
    parameter integer SAMPLE_W = 16,
    parameter integer STATUS_BYTES = 2,
    parameter integer COUNT_W = 8,
    parameter integer WATERMARK = 8,
    parameter integer MAX_BURST_SAMPLES = 32,
    parameter integer PACKET_BYTES = 2,
    parameter integer POLL_PERIOD = 1_000_000,
    parameter integer TIMESTAMP_PER_SAMPLE = 1
)(
    input  wire                  clk,
    input  wire                  resetn,

    input  wire [31:0]           t_now,

    // i2c command interface
    output reg                   i2c_cmd_valid,
    input  wire                  i2c_cmd_ready,
    output reg  [6:0]            i2c_cmd_addr,
    output reg  [7:0]            i2c_cmd_reg,
    output reg  [7:0]            i2c_cmd_len,
    output reg                   i2c_cmd_write,
    output reg  [7:0]            i2c_cmd_wdata,

    // i2c response interface
    input  wire                  i2c_rsp_valid,
    input  wire [7:0]            i2c_rsp_data,
    input  wire                  i2c_rsp_last,
    input  wire                  i2c_rsp_err,
    output reg                   i2c_rsp_ready,

    // output samples
    output reg  [SAMPLE_W-1:0]   ppg_sample,
    output reg                   ppg_sample_valid,
    output reg  [31:0]           ppg_sample_time,

    output reg                   fifo_overflow_flag,
    output reg                   fifo_empty_flag,
    output reg                   i2c_error_flag
);

    localparam integer STATUS_W = 16;
    localparam integer THRESH_W = 16;
    localparam integer SAMPLE_BYTES = (SAMPLE_W + 7) / 8;
    localparam integer PACKET_BYTES_EFF = (PACKET_BYTES <= 0) ? 1 : PACKET_BYTES;
    localparam integer MAX_BURST_BYTES = MAX_BURST_SAMPLES * SAMPLE_BYTES;
    localparam [7:0] FIFO_EN_BITVAL = 8'h01;

    typedef enum logic [3:0] {
        ST_POLL = 4'd0,
        ST_STATUS_CMD = 4'd1,
        ST_STATUS_RECV = 4'd2,
        ST_THRESH_CMD = 4'd3,
        ST_THRESH_RECV = 4'd4,
        ST_DECIDE = 4'd5,
        ST_ENA1_CMD = 4'd6,
        ST_ENA2_CMD = 4'd7,
        ST_DATA_CMD = 4'd8,
        ST_DATA_RECV = 4'd9,
        ST_DIS_CMD = 4'd10
    } state_t;

    state_t state_r;

    reg [31:0] poll_cnt_r;
    reg [STATUS_W-1:0] status_shift_r;
    reg [$clog2(STATUS_BYTES+1)-1:0] status_byte_idx_r;
    reg [THRESH_W-1:0] thresh_shift_r;
    reg [$clog2(3)-1:0] thresh_byte_idx_r;

    reg [SAMPLE_W-1:0] sample_shift_r;
    reg [$clog2(SAMPLE_BYTES+1)-1:0] sample_byte_idx_r;

    reg [15:0] bytes_left_r;
    reg [15:0] read_bytes_r;
    reg [15:0] samples_left_r;
    reg [31:0] burst_time_r;

    wire poll_hit = (poll_cnt_r == (POLL_PERIOD - 1));

    wire overflow_w = status_shift_r[7];
    wire [7:0] fifo_bytes_avail_w = status_shift_r[15:8];
    wire [5:0] fifo_thresh_words_w = thresh_shift_r[13:8];
    wire [5:0] fifo_thresh_words_eff_w = (fifo_thresh_words_w != 6'd0) ?
                                         fifo_thresh_words_w :
                                         ((WATERMARK > 63) ? 6'd63 : WATERMARK[5:0]);
    wire [15:0] fifo_thresh_bytes_w = {9'd0, fifo_thresh_words_eff_w, 1'b0};
    wire fifo_empty_w = (fifo_bytes_avail_w == 8'd0);

    wire [15:0] fifo_bytes_avail_ext_w = {8'd0, fifo_bytes_avail_w};
    wire [15:0] read_bytes_pre_w =
        (fifo_bytes_avail_ext_w > MAX_BURST_BYTES[15:0]) ? MAX_BURST_BYTES[15:0] : fifo_bytes_avail_ext_w;
    wire [15:0] read_bytes_pkt_w = (read_bytes_pre_w / PACKET_BYTES_EFF) * PACKET_BYTES_EFF;
    wire [15:0] read_samples_w = read_bytes_pkt_w / SAMPLE_BYTES;
    wire should_read_w = (fifo_bytes_avail_ext_w >= fifo_thresh_bytes_w) &&
                         (read_bytes_pkt_w >= PACKET_BYTES_EFF);

    always @(posedge clk) begin
        if (!resetn) begin
            state_r <= ST_POLL;
            poll_cnt_r <= 32'd0;

            i2c_cmd_valid <= 1'b0;
            i2c_cmd_addr <= I2C_ADDR;
            i2c_cmd_reg <= 8'h00;
            i2c_cmd_len <= 8'd0;
            i2c_cmd_write <= 1'b0;
            i2c_cmd_wdata <= 8'd0;

            i2c_rsp_ready <= 1'b0;

            status_shift_r <= {STATUS_W{1'b0}};
            status_byte_idx_r <= '0;
            thresh_shift_r <= {THRESH_W{1'b0}};
            thresh_byte_idx_r <= '0;

            sample_shift_r <= {SAMPLE_W{1'b0}};
            sample_byte_idx_r <= '0;
            bytes_left_r <= 16'd0;
            read_bytes_r <= 16'd0;
            samples_left_r <= 16'd0;
            burst_time_r <= 32'd0;

            ppg_sample <= {SAMPLE_W{1'b0}};
            ppg_sample_valid <= 1'b0;
            ppg_sample_time <= 32'd0;

            fifo_overflow_flag <= 1'b0;
            fifo_empty_flag <= 1'b0;
            i2c_error_flag <= 1'b0;
        end else begin
            ppg_sample_valid <= 1'b0;
            i2c_cmd_valid <= 1'b0;
            i2c_cmd_write <= 1'b0;
            i2c_cmd_wdata <= 8'd0;
            i2c_rsp_ready <= 1'b0;

            case (state_r)
                ST_POLL: begin
                    if (poll_hit) begin
                        poll_cnt_r <= 32'd0;
                        state_r <= ST_STATUS_CMD;
                    end else begin
                        poll_cnt_r <= poll_cnt_r + 32'd1;
                    end
                end

                ST_STATUS_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_STATUS;
                    i2c_cmd_len <= 8'd2;
                    if (i2c_cmd_ready) begin
                        status_shift_r <= {STATUS_W{1'b0}};
                        status_byte_idx_r <= '0;
                        state_r <= ST_STATUS_RECV;
                    end
                end

                ST_STATUS_RECV: begin
                    i2c_rsp_ready <= 1'b1;
                    if (i2c_rsp_valid) begin
                        if (i2c_rsp_err) begin
                            i2c_error_flag <= 1'b1;
                            state_r <= ST_POLL;
                        end else begin
                            begin
                                reg [STATUS_W-1:0] status_next;
                                status_next = status_shift_r | ({{(STATUS_W-8){1'b0}}, i2c_rsp_data} << (status_byte_idx_r * 8));
                                status_shift_r <= status_next;
                                status_byte_idx_r <= status_byte_idx_r + 1'b1;
                                if ((status_byte_idx_r == (STATUS_BYTES-1)) || i2c_rsp_last) begin
                                    fifo_overflow_flag <= fifo_overflow_flag | status_next[7];
                                    fifo_empty_flag <= (status_next[15:8] == 8'd0);
                                    state_r <= ST_THRESH_CMD;
                                end
                            end
                        end
                    end
                end

                ST_THRESH_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_FIFO_THRESH;
                    i2c_cmd_len <= 8'd2;
                    if (i2c_cmd_ready) begin
                        thresh_shift_r <= {THRESH_W{1'b0}};
                        thresh_byte_idx_r <= '0;
                        state_r <= ST_THRESH_RECV;
                    end
                end

                ST_THRESH_RECV: begin
                    i2c_rsp_ready <= 1'b1;
                    if (i2c_rsp_valid) begin
                        if (i2c_rsp_err) begin
                            i2c_error_flag <= 1'b1;
                            state_r <= ST_POLL;
                        end else begin
                            begin
                                reg [THRESH_W-1:0] thresh_next;
                                thresh_next = thresh_shift_r | ({{(THRESH_W-8){1'b0}}, i2c_rsp_data} << (thresh_byte_idx_r * 8));
                                thresh_shift_r <= thresh_next;
                                thresh_byte_idx_r <= thresh_byte_idx_r + 1'b1;
                                if ((thresh_byte_idx_r == 2-1) || i2c_rsp_last) begin
                                    state_r <= ST_DECIDE;
                                end
                            end
                        end
                    end
                end

                ST_DECIDE: begin
                    if (should_read_w) begin
                        read_bytes_r <= read_bytes_pkt_w;
                        bytes_left_r <= read_bytes_pkt_w;
                        samples_left_r <= read_samples_w;
                        sample_shift_r <= {SAMPLE_W{1'b0}};
                        sample_byte_idx_r <= '0;
                        if (TIMESTAMP_PER_SAMPLE == 0) burst_time_r <= t_now;
                        state_r <= ST_ENA1_CMD;
                    end else begin
                        state_r <= ST_POLL;
                    end
                end

                ST_ENA1_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_FIFO_ACCESS_ENA;
                    i2c_cmd_len <= 8'd1;
                    i2c_cmd_write <= 1'b1;
                    i2c_cmd_wdata <= FIFO_EN_BITVAL;
                    if (i2c_cmd_ready) begin
                        state_r <= ST_ENA2_CMD;
                    end
                end

                ST_ENA2_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_FIFO_ACCESS_ENA;
                    i2c_cmd_len <= 8'd1;
                    i2c_cmd_write <= 1'b1;
                    i2c_cmd_wdata <= FIFO_EN_BITVAL;
                    if (i2c_cmd_ready) begin
                        state_r <= ST_DATA_CMD;
                    end
                end

                ST_DATA_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_FIFO_ACCESS;
                    i2c_cmd_len <= read_bytes_r[7:0];
                    if (i2c_cmd_ready) begin
                        state_r <= ST_DATA_RECV;
                    end
                end

                ST_DATA_RECV: begin
                    i2c_rsp_ready <= 1'b1;
                    if (i2c_rsp_valid) begin
                        if (i2c_rsp_err) begin
                            i2c_error_flag <= 1'b1;
                            state_r <= ST_DIS_CMD;
                        end else begin
                            sample_shift_r <= sample_shift_r | ({{(SAMPLE_W-8){1'b0}}, i2c_rsp_data} << (sample_byte_idx_r * 8));
                            sample_byte_idx_r <= sample_byte_idx_r + 1'b1;

                            if ((sample_byte_idx_r == (SAMPLE_BYTES-1)) || i2c_rsp_last) begin
                                ppg_sample <= sample_shift_r | ({{(SAMPLE_W-8){1'b0}}, i2c_rsp_data} << (sample_byte_idx_r * 8));
                                ppg_sample_valid <= 1'b1;
                                if (TIMESTAMP_PER_SAMPLE != 0) ppg_sample_time <= t_now;
                                else ppg_sample_time <= burst_time_r;

                                sample_shift_r <= {SAMPLE_W{1'b0}};
                                sample_byte_idx_r <= '0;

                                if (samples_left_r > 0) samples_left_r <= samples_left_r - 1'b1;
                            end

                            if (bytes_left_r > 0) bytes_left_r <= bytes_left_r - 1'b1;
                            if ((bytes_left_r == 16'd1) || i2c_rsp_last) begin
                                state_r <= ST_DIS_CMD;
                            end
                        end
                    end
                end

                ST_DIS_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_FIFO_ACCESS_ENA;
                    i2c_cmd_len <= 8'd1;
                    i2c_cmd_write <= 1'b1;
                    i2c_cmd_wdata <= 8'h00;
                    if (i2c_cmd_ready) begin
                        state_r <= ST_POLL;
                        bytes_left_r <= 16'd0;
                        read_bytes_r <= 16'd0;
                        samples_left_r <= 16'd0;
                        sample_shift_r <= {SAMPLE_W{1'b0}};
                        sample_byte_idx_r <= '0;
                    end
                end

                default: state_r <= ST_POLL;
            endcase
        end
    end

endmodule
