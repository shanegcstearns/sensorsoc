module top #() (
    input clk_i
    ,input reset_i
    


    ,output alarm_o

);
    i2c_master #() i2c_mast (
        .clk(clk_i),
        .resetn(reset_i),
        // Functional I2C bus :: Are these all inputs/outputs in top? isnt I2C 2 wires
        .i2c_req, //o
        .i2c_addr, //o 7b
        .i2c_reg, //o 8b
        .i2c_len, //o 8b
        .i2c_ack, //i
        .i2c_rdata, //i 8b
        .i2c_rvalid, //i
        .i2c_rlast, //i
        .i2c_err, //i
        // Accel output
        .accel_valid, //o
        .accel_ax, //o 14b
        .accel_ay, //o 14b
        .accel_az, //o 14b

        // PPG output
        .ppg_valid, //o
        .ppg_red, //o 14b ppg read output
        .ppg_ir, //o 14b ppg timestamp(?)

        // Status
        .accel_err_o, //o
        .ppg_err_o, //o

        // Config
        .enable_i //i
    );

    accel_reader #() accel_read (
        .clk(clk_i),
        .resetn(reset_i),

        .cfg_enable_i(1'b1), //this is an overall enable?
        .cfg_init_en_i(1'b1), //i2c init sequence enable, should this be tied to 1?
        .cfg_poll_period_ticks_i(32'h000186A0), //polling period for accel reader, dependent on clock? so if 10MHz clk with 100Hz samples from sensor, 100_000 polling period? 
        .cfg_ctrl1_data_i, //8bitsm what should these be, ananya question?
        .cfg_range_data_i, //^^

        .t_now_i(), //timestamp from global timer

        // I2C command interface
        .i2c_cmd_valid_o(), //i have a command to read ready
        .i2c_cmd_ready_i(), //input saying i2c master is ready to receive command
        .i2c_cmd_addr_o(), //output address for i2c 7bit
        .i2c_cmd_reg_o(), //register access, prob ignore
        .i2c_cmd_len_o(), //number of bytes to read/write
        .i2c_cmd_write_o(), //i2c command 1 == write, 0 == read
        .i2c_cmd_wdata_o(), //data to be written on write commands

        // I2C response interface
        .i2c_rsp_valid_i(), //valid
        .i2c_rsp_data_i(), //8bit data written back
        .i2c_rsp_done_i(), //if resp is done
        .i2c_rsp_error_i(), //was there an error

        // Output samples
        .ax_o(), //16b 
        .ay_o(), //16b
        .az_o(), //16b 
        .accel_valid_o(), //data valid
        .accel_sample_time_o(), //time output on complete data

        .init_done_o(), //if init was done successfully, prob means we can pull cfg_init_en_i low when this is high
        .i2c_error_o(), //errors on: NACK seen, Unexpected length, Timeout
        .timeout_o(), //if we hit timeout resp
        .nack_seen_o(), //if theres a nack
    );

    ppg_fifo_reader #() ppg_read (
        .clk(clk_i),
        .resetn(reset_i),

        .t_now(), //global timestamp 32b

        // i2c command interface
        .i2c_cmd_valid(), //i have a command to read ready
        .i2c_cmd_ready(), //input saying i2c master is ready to receive command
        .i2c_cmd_addr(), //output address for i2c 7bit
        .i2c_cmd_reg(), //register access, prob ignore
        .i2c_cmd_len(), //number of bytes to read/write
        .i2c_cmd_write(), //i2c command 1 == write, 0 == read
        .i2c_cmd_wdata(), //data to be written on write commands

        // i2c response interface
        .i2c_rsp_valid(), //i resp valid signal
        .i2c_rsp_data(), //i 8b resp data byte
        .i2c_rsp_last(), //i last bit signal
        .i2c_rsp_err(), //resp error signal
        .i2c_rsp_ready(), //resp ready

        // output samples
        .ppg_sample(), //o 16b data output
        .ppg_sample_valid(), //o is the sample valid currently
        .ppg_sample_time(), //o 32b timestamp from the data

        .fifo_overflow_flag(), //fifo full
        .fifo_empty_flag(), //fifo empty
        .i2c_error_flag(), //fifo error
    );

    ppg_beat_detect_rr_calc #() beat_detect (
        .clk_i,
        .rst_ni(reset_i),

        .ppg_sample_i(), //16b ppg sample input
        .ppg_valid_i(), //ppg valid
        .ppg_sample_time_i, //ppg timestamp inp
        .timebase_i, //global timer input prob ignore with config

        .cfg_enable_i(1'b1), //module enable 
        .cfg_bypass_i(1'b0), //detection log bypass
        .cfg_time_src_i(1'b1),       // 0: timebase_i, 1: ppg_sample_time_i
        .cfg_signed_i(1'b0),         // 0: unsigned sample, 1: signed sample, i think they are all unsigned?

        .cfg_lp_beta_i(), //beta coeff for math coming from MMIO 12bit
        .cfg_base_alpha_i(), //same but for alpha 12bit
        .cfg_env_decay_i(), //24b how fast envelope decays? MMIO
        .cfg_abs_en_i(1'b1), //use absolute val, i think this should be 1

        .cfg_thr_k_i(), //scaling factor for thresholds, 12bit also in MMIO
        .cfg_thr_min_i(), //min threshold floor? 24b

        input  wire [T_W-1:0]               cfg_refrac_ticks_i,
        input  wire [T_W-1:0]               cfg_rr_min_ticks_i,
        input  wire [T_W-1:0]               cfg_rr_max_ticks_i,
        input  wire                         cfg_peak_mode_i,      // 0: local-max, 1: rising-edge

        input  wire [7:0]                   cfg_q_amp_w_i,
        input  wire [7:0]                   cfg_q_slope_w_i,
        input  wire [7:0]                   cfg_q_refrac_penalty_i,
        input  wire [7:0]                   cfg_q_min_accept_i,

        output reg                          beat_pulse_o,
        output reg                          rr_valid_o,
        output reg                          rr_accepted_o,
        output reg  [T_W-1:0]               rr_interval_o,
        output reg  signed [16:0]           delta_hr_bpm_o,

        output reg  [7:0]                   beat_quality_o,
        output reg                          double_beat_o,
        output reg                          missed_beat_o,
        output reg                          ppg_invalid_o
    );


endmodule