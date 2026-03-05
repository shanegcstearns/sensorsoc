module top #() (
    input clk_i
    ,input reset_i
    ,//TODO: are we using current i2c master? what are the IO pins EXACTLY?
    ,output alarm_o

);
    // converting seconds to ms
    localparam int unsigned CLK_HZ = 10_000_000;
    localparam int unsigned MS_DIV = (CLK_HZ >= 1000) ? (CLK_HZ / 1000) : 1;
    localparam int unsigned MS_DIV_W = (MS_DIV <= 1) ? 1 : $clog2(MS_DIV);

    logic [MS_DIV_W-1:0] ms_div_q;
    logic [31:0] time_ms_w;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            ms_div_q  <= '0;
            time_ms_w <= 32'd0;
        end else if (ms_div_q == MS_DIV-1) begin
            ms_div_q  <= '0;
            time_ms_w <= time_ms_w + 32'd1;
        end else begin
            ms_div_q <= ms_div_q + 1'b1;
        end
    end

    i2c_master #() i2c_mast (
        .clk(clk_i),
        .resetn(~reset_i),
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
        .enable_i(2'b11) //i
    );
    logic [15:0] seconds_w;
    logic epoch_end_w;

    globaltimer #() glob_time (
        .clk_i(clk_i),
        .rst_i(reset_i),
        .en_i(1'b1), // should go high and stay high as soon as the "night" starts, 1 for now
        .time_in_night_seconds_o(seconds_w), //time in seconds   // TODO: should be 32 bits for unix time     
        .epoch_end_o(epoch_end_w),                      // high for 1 cycle after 1000 cycles (10 seconds at 100 Hz)
        .epoch_index_o()                     // which epoch we are on (0 to 999, wraps around)
    ); 

    logic [15:0] cos_w;

    cos_lut_timer #() cos_feat (
        .clk_i(clk_i),
        .rst_i(reset_i),

        .cfg_enable_i(1'b1), //enable, 1 for now
        .seconds_in_night_i({16'h0000, seconds_w}),
        .seconds_valid_i(1'b1), //timer should always be valid, unless reset is high
        .cfg_period_seconds_i(32'h00015180), //total nighttime, thats number of seconds in a day, might need to change how to calc feature
        .cfg_lut_bits_i(3'd6),   //  setting LUT size
        .cfg_scale_q15_i(16'h7FFF),  // 1.0 -> 32767, should be 3.13, no scale

        .cos_time_feat_o(cos_w)
    );

    // TODO: replace these with real connections once i2c master is done
    // Accel reader cmd outputs
    logic        acc_cmd_ready_w;
    logic        acc_rsp_valid_w;
    logic [7:0]  acc_rsp_data_w;
    logic        acc_rsp_done_w;
    logic        acc_rsp_error_w;

    assign acc_cmd_ready_w  = 1'b0;
    assign acc_rsp_valid_w  = 1'b0;
    assign acc_rsp_data_w   = 8'h00;
    assign acc_rsp_done_w   = 1'b0;
    assign acc_rsp_error_w  = 1'b0;

    logic [15:0] ax_w, ay_w, az_w; 
    logic accel_valid_w, accel_error_w;

    accel_reader #() accel_read (
        .clk(clk_i),
        .rst_i(reset_i),

        .cfg_enable_i(1'b1), // overall enable
        .cfg_init_en_i(1'b1), // enables config writes to accel regs at startup
        .cfg_poll_period_ticks_i(32'd50000), // poll period in clk ticks (5MHz/100Hz=50,000)
        .cfg_ctrl1_data_i(8'h57), // 100 Hz ODR, XYZ enable (LIS2DW/LIS3 class default style)
        .cfg_range_data_i(8'h00), // +/-2g full scale

        // I2C command interface
        .i2c_cmd_valid_o(), //i have a command to read ready
        .i2c_cmd_ready_i(acc_cmd_ready_w), //input saying i2c master is ready to receive command
        .i2c_cmd_addr_o(), //output address for i2c 7bit
        .i2c_cmd_reg_o(), //register access, prob ignore
        .i2c_cmd_len_o(), //number of bytes to read/write
        .i2c_cmd_write_o(), //i2c command 1 == write, 0 == read
        .i2c_cmd_wdata_o(), //data to be written on write commands

        // I2C response interface
        .i2c_rsp_valid_i(acc_rsp_valid_w), //valid
        .i2c_rsp_data_i(acc_rsp_data_w), //8bit data written back
        .i2c_rsp_done_i(acc_rsp_done_w), //if resp is done
        .i2c_rsp_error_i(acc_rsp_error_w), //was there an error

        // Output samples
        .ax_o(ax_w), //16b 
        .ay_o(ay_w), //16b
        .az_o(az_w), //16b 
        .accel_valid_o(accel_valid_w), //data valid

        // DEBUG SIGNALS
        .init_done_o(), // debug signal to check if init successful
        .i2c_error_o(accel_error_w), //errors on: NACK seen, Unexpected length, Timeout
        .timeout_o(), //if we hit timeout resp
        .nack_seen_o(), //if theres a nack 
    );

    logic [47:0] motion_mag_w; 
    logic motion_epoch_w;

    motion_preprocess #(
        .AX_W(16)
    ) motion_prepros ( 
        .clk(clk_i),
        .rst_i(reset_i),
        // Sample capture
        .sample_valid_i(accel_valid_w), //sample in valid
        .sample_ok_i(!accel_error_w), //sample is good data
        .ax_i(ax_w), //14b 
        .ay_i(ay_w), //14b 
        .az_i(az_w), //14b
        // Epoch control (external)
        .epoch_end_i(epoch_end_w), //input saying when epoch is done
        // Per epoch outputs (latched on epoch end)
        .epoch_done_o(motion_epoch_w), //passing on epoch end_i, with latching?
        .motion_energy_epoch_o(motion_mag_w) //48b total movement
    );

    logic [31:0] ppg_timestamp_w;
    logic [15:0] ppg_sample_w;
    logic ppg_valido_w, fifo_over_w, fifo_empty_w, ppg_i2c_err_w;

    // TODO: replace these with real connections once i2c master is done
    // PPG FIFO reader rsp inputs
    logic        ppg_cmd_ready_w;
    logic        ppg_rsp_valid_w;
    logic [7:0]  ppg_rsp_data_w;
    logic        ppg_rsp_last_w;
    logic        ppg_rsp_err_w;
    logic        ppg_rsp_ready_w;

    assign ppg_cmd_ready_w  = 1'b0;
    assign ppg_rsp_valid_w  = 1'b0;
    assign ppg_rsp_data_w   = 8'h00;
    assign ppg_rsp_last_w   = 1'b0;
    assign ppg_rsp_err_w    = 1'b0;

    ppg_fifo_reader #() ppg_read (
        .clk_i(clk_i),
        .rst_i(reset_i),

        .t_now(time_ms_w), //global timestamp in ms

        // i2c command interface
        .i2c_cmd_valid(), //i have a command to read ready
        .i2c_cmd_ready(ppg_cmd_ready_w), //input saying i2c master is ready to receive command
        .i2c_cmd_addr(), //output address for i2c 7bit
        .i2c_cmd_reg(), //register access, prob ignore
        .i2c_cmd_len(), //number of bytes to read/write
        .i2c_cmd_write(), //i2c command 1 == write, 0 == read
        .i2c_cmd_wdata(), //data to be written on write commands

        // i2c response interface
        .i2c_rsp_valid(ppg_rsp_valid_w), //i resp valid signal
        .i2c_rsp_data(ppg_rsp_data_w), //i 8b resp data byte
        .i2c_rsp_last(ppg_rsp_last_w), //i last bit signal
        .i2c_rsp_err(ppg_rsp_err_w), //resp error signal
        .i2c_rsp_ready(ppg_rsp_ready_w), //resp ready

        // output samples
        .ppg_sample(ppg_sample_w), //o 16b data output
        .ppg_sample_valid(ppg_valido_w), //o is the sample valid currently
        .ppg_sample_time(ppg_timestamp_w), //o 32b timestamp from the data

        // go into signal quality module
        .fifo_overflow_flag(fifo_over_w), //fifo full
        .fifo_empty_flag(fifo_empty_w), //fifo empty
        .i2c_error_flag(ppg_i2c_err_w), //fifo error
    );

    // Beat detect defaults
    localparam logic [11:0] CFG_LP_BETA_Q10      = 12'd128;  // 0.125
    localparam logic [11:0] CFG_BASE_ALPHA_Q10   = 12'd16;   // 0.015625
    localparam logic [23:0] CFG_ENV_DECAY         = 24'd8;
    localparam logic [11:0] CFG_THR_K_Q10         = 12'd512;  // 0.5
    localparam logic [23:0] CFG_THR_MIN           = 24'd32;
    localparam logic [31:0] CFG_REFRACT_MS        = 32'd250;
    localparam logic [31:0] CFG_RR_MIN_MS         = 32'd300;
    localparam logic [31:0] CFG_RR_MAX_MS         = 32'd2000;
    localparam logic [7:0]  CFG_Q_AMP_W           = 8'd4;
    localparam logic [7:0]  CFG_Q_SLOPE_W         = 8'd2;
    localparam logic [7:0]  CFG_Q_REFRAC_PENALTY  = 8'd24;
    localparam logic [7:0]  CFG_Q_MIN_ACCEPT      = 8'd10;

    // Signal quality defaults
    localparam logic [7:0]  CFG_BEAT_Q_MIN        = 8'd16;
    localparam logic [7:0]  CFG_MIN_VALID_FRAC    = 8'd96;    // about 38%
    localparam logic [7:0]  CFG_MAX_DOUBLE        = 8'd4;
    localparam logic [7:0]  CFG_MAX_MISSED        = 8'd3;
    localparam logic [15:0] CFG_MOTION_HI_TH      = 16'd2000;
    localparam logic [15:0] CFG_MAX_MOTION_HI     = 16'd3;

    logic [31:0] rr_int_w;
    logic signed [16:0] delta_hr_w;
    logic [7:0] beat_qual_w;
    logic rr_accepted_w, rr_valid_w, double_beat_w, missed_beat_w, ppg_invalid_w, beat_pulse_w;

    ppg_beat_detect_rr_calc #() beat_detect ( // in from ppg reader, out to RMSSD and feature engine
        .clk_i,
        .rst_i(reset_i),

        .ppg_sample_i(ppg_sample_w), //16b ppg sample input
        .ppg_valid_i(ppg_valido_w), //ppg valid
        .ppg_sample_time_i(ppg_timestamp_w), //32b ppg reader timestamp inp

        .cfg_enable_i(1'b1), //module enable 
        .cfg_bypass_i(1'b0), //detection log bypass
        .cfg_signed_i(1'b0),         // 0: unsigned sample, 1: signed sample, i think they are all unsigned?

        .cfg_lp_beta_i(CFG_LP_BETA_Q10), // low-pass filter coefficient for smoothing
        .cfg_base_alpha_i(CFG_BASE_ALPHA_Q10), // baseline tracking coefficient for high pass removal
        .cfg_env_decay_i(CFG_ENV_DECAY), // envelope decay (for adaptive thresholding)
        .cfg_abs_en_i(1'b1), //use absolute val

        .cfg_thr_k_i(CFG_THR_K_Q10), //scaling factor for thresholds, Q10
        .cfg_thr_min_i(CFG_THR_MIN), //min threshold floor (prevents too low threshold)

        .cfg_refrac_ticks_i(CFG_REFRACT_MS), // refractory period in ms
        .cfg_rr_min_ticks_i(CFG_RR_MIN_MS), // reject RR below this (ms)
        .cfg_rr_max_ticks_i(CFG_RR_MAX_MS), // reject RR above this (ms)
        .cfg_peak_mode_i(1'b0),      // 0: local-max, 1: rising-edge

        //values for scoring beat amounts
        .cfg_q_amp_w_i(CFG_Q_AMP_W), //8b amplitude weight
        .cfg_q_slope_w_i(CFG_Q_SLOPE_W), //8b slope weight
        .cfg_q_refrac_penalty_i(CFG_Q_REFRAC_PENALTY), //penalty if too close to refractory
        .cfg_q_min_accept_i(CFG_Q_MIN_ACCEPT), //min score to allow a beat reading

        .beat_pulse_o(beat_pulse_w), //heartbeat signal
        .rr_valid_o(rr_valid_w), //pulse when RR interval is done
        .rr_accepted_o(rr_accepted_w), //interval was valid throughout
        .rr_interval_o(rr_int_w), //32b 'epoch'/ interval for RR
        .delta_hr_bpm_o(delta_hr_w), //16b delta_hr assumes timestamps are in milliseconds, shane has to change global or nithin changes this

        .beat_quality_o(beat_qual_w), //8b quality score output (from above)
        .double_beat_o(double_beat_w), //RR too short, double beat detection (under min ticks)
        .missed_beat_o(missed_beat_w), //RR too long, missed a beat (over max ticks)
        .ppg_invalid_o(ppg_invalid_w) //invalid, interval too long
    );

    logic [31:0] rmssd_w;
    logic rmssd_valid_w;

    rmssd_engine #() rmssd_feat (
        .clk_i,
        .rst_i(reset_i),

        .rr_interval_i(rr_int_w), //32b rr interval from beat detect
        .rr_valid_i(rr_valid_w), //rr valid
        .rr_accepted_i(rr_accepted_w), //rr accepted  
        .epoch_end_i(epoch_end_w), //epoch end from global timer

        .rmssd_epoch_o(rmssd_w), //32b rmssd per epoch
        .rmssd_valid_o(rmssd_valid_w), //valid_o signal
        .rr_diff_count_o() //16b how many rr differences contributed to the rmssd
    );

    logic valid_feats_w;

    signal_quality #() signal_qual (
        .clk_i(clk_i),
        .rst_i(reset_i),

        .epoch_end_i(epoch_end_w), //epoch end from global timer

        .beat_event_i(beat_pulse_w), // beat event 
        .beat_quality_i(beat_qual_w), //8b beat quality from beat detect
        .double_beat_i(double_beat_w), //double beat detect
        .missed_beat_i(missed_beat_w), //missed beat

        .fifo_overflow_i(fifo_over_w), //fifo overflow from ppg fifo reader
        .fifo_i2c_error_i(ppg_i2c_err_w), //fifo error from ppg fifo reader

        .motion_valid_i(motion_epoch_w), // motion epoch summary is valid
        .motion_intensity_i(motion_mag_w[47:32]), //16b motion input, maybe just take top 16b

        .cfg_beat_q_min_i(CFG_BEAT_Q_MIN), //8b beat quality minimum
        .cfg_min_valid_fraction_i(CFG_MIN_VALID_FRAC), //8b minimum fraction of valid beats
        .cfg_max_double_i(CFG_MAX_DOUBLE), //8b max number of double heartbeat detections allowed
        .cfg_max_missed_i(CFG_MAX_MISSED), //8b max number of missed heartbeats detected
        .cfg_motion_hi_th_i(CFG_MOTION_HI_TH), //16b threshold for movement that's too high
        .cfg_max_motion_hi_i(CFG_MAX_MOTION_HI), //16b max high-motion epochs

        .invalid_reason_o(), //8b 
        /*
        1. fifo overflow
        2. i2c err occured
        3. no beats detected
        4. too low ratio of valid/invalid beats
        5. too many double beats
        6. too many missing beats
        7. too many maxed out motion detects
        8. empty for now
        */
        .ml_update_gate_o(valid_feats_w) //tells feat engine if data is good to go
    );

    logic [15:0] motion_ml_w, cos_ml_w, delta_ml_w, rmssd_ml_w; 
    logic feat_eng_valid_w;

    feature_engine feat_en(
        .clk_i(clk_i),
        .rst_i(reset_i),
        .enable_i(1'b1), //module enable
        .seconds_valid_i(1'b1), //cos LUT inp valid (always valid)
        .cos_time_feat_i(cos_w), //16b cos time feat from LUT
        // motion
        .motion_valid_i(motion_epoch_w), //motion preprocessing valid
        .motion_energy_epoch_i(motion_mag_w),//16b motion 'mag' feature

        // delta hr
        .delta_hr_valid_i(rr_valid_w), //delta hr valid from beat detect
        .delta_hr_i(delta_hr_w[15:0]), //delta hr feature

        // rmssd
        .rmssd_valid_i(rmssd_valid_w), //rmssd valid signal
        .rmssd_i(rmssd_w[15:0]), //16b rmssd feature

        // valid
        .feat_valid_o(feat_eng_valid_w), //feature valid_o
        .time_feat_o(cos_ml_w), //16b time feat to ml
        .motion_feat_o(motion_ml_w), //16b motion feat to ml
        .delta_hr_feat_o(delta_ml_w), //16b delta_hr to ml
        .rmssd_feat_o(rmssd_ml_w), //16b rmssd to ml
        
        // ML gate from signal quality
        .ml_update_gate_i(valid_feats_w) //feat quality update sig from sig_qual module
    );

    axi_interface feats_to_ml(
        .CLK(clk_i),
        .RESETN(~reset_i),

        // control
        .start(feat_eng_valid_w), //this should come from the CPU? or maybe just the valid signal
        .x_addr(), //TODO also from cpu, the base offsets that were given from ml model
        .x0(motion_ml_w),     // movement
        .x1(cos_ml_w),     // cosine
        .x2(delta_ml_w),     // delta_hr
        .x3(rmssd_ml_w),     // rmssd
        .done,          // high when writes completed
        .busy,          // high while writing

        // AXI4 write address channel
        .maxi_awaddr(), //o 32b address write output?
        .maxi_awlen(), //o 8b address write len
        .maxi_awsize(), //o 3b
        .maxi_awburst(), //o 2b
        .maxi_awlock(), //o
        .maxi_awcache(), //o 4b
        .maxi_awprot(), //o 3b
        .maxi_awqos(), //o 4b
        .maxi_awuser(), //o 2b
        .maxi_awvalid(), //o
        .maxi_awready(), //i

        // AXI4 write data channel
        .maxi_wdata(), //o 32b 
        .maxi_wstrb(), //o 4b
        .maxi_wlast(), //o
        .maxi_wvalid(), //o
        .maxi_wready(), //i

        // AXI4 write response channel
        .maxi_bresp(), //i 2b
        .maxi_bvalid(), //i
        .maxi_bready() //o
    );

endmodule
