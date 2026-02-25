module ppg_file_player #(
//    parameter string FILE = "C:\Users\amand\CSE127A\sensorsoc\sensor_models\sensor_output\ppg_digital.csv",
    parameter int FS_HZ = 1
)(
    input  logic clk,
    input  logic resetn,

    output logic sample_valid,
    output logic signed [15:0] sample
);

    int fd;
    int r;

    int divider = 50_000_000 / FS_HZ;
    int cnt;

    initial begin
        fd = $fopen("sim/data/ppg_digital.csv", "r");
        if (fd == 0) begin
            $display("ERROR: Could not open ppg CSV");
            $fatal(1);
        end
    end

    always @(posedge clk) begin
        if (!resetn) begin
            cnt <= 0;
            sample_valid <= 0;
        end else begin
            cnt <= cnt + 1;

            if (cnt >= divider) begin
                cnt <= 0;

                r = $fscanf(fd, "%d\n", sample);

                if (r == 1)
                    sample_valid <= 1;
                else
                    sample_valid <= 0;
            end else begin
                sample_valid <= 0;
            end
        end
    end

endmodule
