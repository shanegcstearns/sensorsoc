`timescale 1ns/1ps

module ppg_fifo_reader #(
    parameter [6:0]  I2C_ADDR = 7'h64,
    parameter [7:0]  REG_STATUS = 8'h06,
    parameter [7:0]  REG_DATA = 8'h08,
    parameter integer SAMPLE_W = 16,
    parameter integer STATUS_BYTES = 2,
    parameter integer COUNT_W = 8,
    parameter integer WATERMARK = 8,
    parameter integer MAX_BURST_SAMPLES = 32,
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

    localparam integer STATUS_W = STATUS_BYTES * 8;
    localparam integer SAMPLE_BYTES = (SAMPLE_W + 7) / 8;

    typedef enum logic [2:0] {
        ST_POLL = 3'd0,
        ST_STATUS_CMD = 3'd1,
        ST_STATUS_RECV = 3'd2,
        ST_DECIDE = 3'd3,
        ST_DATA_CMD = 3'd4,
        ST_DATA_RECV = 3'd5
    } state_t;

    state_t state_r;

    reg [31:0] poll_cnt_r;
    reg [STATUS_W-1:0] status_shift_r;
    reg [$clog2(STATUS_BYTES+1)-1:0] status_byte_idx_r;

    reg [SAMPLE_W-1:0] sample_shift_r;
    reg [$clog2(SAMPLE_BYTES+1)-1:0] sample_byte_idx_r;

    reg [15:0] samples_left_r;
    reg [15:0] burst_samples_r;
    wire [15:0] burst_bytes_w = burst_samples_r * SAMPLE_BYTES;
    reg [31:0] burst_time_r;

    wire poll_hit = (poll_cnt_r == (POLL_PERIOD - 1));

    wire [STATUS_W-1:0] status_comb = status_shift_r;
    wire overflow_w = status_comb[STATUS_W-1];
    wire [COUNT_W-1:0] count_w = status_comb[COUNT_W-1:0];
    wire fifo_empty_w = (count_w == {COUNT_W{1'b0}});

    wire [15:0] count_ext = {{(16-COUNT_W){1'b0}}, count_w};
    wire [15:0] burst_sel = (count_ext > MAX_BURST_SAMPLES[15:0]) ? MAX_BURST_SAMPLES[15:0] : count_ext;

    always @(posedge clk) begin
        if (!resetn) begin
            state_r <= ST_POLL;
            poll_cnt_r <= 32'd0;

            i2c_cmd_valid <= 1'b0;
            i2c_cmd_addr <= I2C_ADDR;
            i2c_cmd_reg <= 8'h00;
            i2c_cmd_len <= 8'd0;

            i2c_rsp_ready <= 1'b0;

            status_shift_r <= {STATUS_W{1'b0}};
            status_byte_idx_r <= '0;

            sample_shift_r <= {SAMPLE_W{1'b0}};
            sample_byte_idx_r <= '0;
            samples_left_r <= 16'd0;
            burst_samples_r <= 16'd0;
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
                    i2c_cmd_len <= STATUS_BYTES[7:0];
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
                                    fifo_overflow_flag <= fifo_overflow_flag | status_next[STATUS_W-1];
                                    fifo_empty_flag <= (status_next[COUNT_W-1:0] == {COUNT_W{1'b0}});
                                    state_r <= ST_DECIDE;
                                end
                            end
                        end
                    end
                end

                ST_DECIDE: begin
                    if (count_ext >= WATERMARK[15:0]) begin
                        burst_samples_r <= burst_sel;
                        samples_left_r <= burst_sel;
                        state_r <= ST_DATA_CMD;
                    end else begin
                        state_r <= ST_POLL;
                    end
                end

                ST_DATA_CMD: begin
                    i2c_cmd_valid <= 1'b1;
                    i2c_cmd_addr <= I2C_ADDR;
                    i2c_cmd_reg <= REG_DATA;
                    i2c_cmd_len <= burst_bytes_w[7:0];
                    if (i2c_cmd_ready) begin
                        sample_shift_r <= {SAMPLE_W{1'b0}};
                        sample_byte_idx_r <= '0;
                        if (TIMESTAMP_PER_SAMPLE == 0) burst_time_r <= t_now;
                        state_r <= ST_DATA_RECV;
                    end
                end

                ST_DATA_RECV: begin
                    i2c_rsp_ready <= 1'b1;
                    if (i2c_rsp_valid) begin
                        if (i2c_rsp_err) begin
                            i2c_error_flag <= 1'b1;
                            state_r <= ST_POLL;
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
                                if ((samples_left_r == 16'd1) || i2c_rsp_last) begin
                                    state_r <= ST_POLL;
                                end
                            end
                        end
                    end
                end

                default: state_r <= ST_POLL;
            endcase
        end
    end

endmodule
