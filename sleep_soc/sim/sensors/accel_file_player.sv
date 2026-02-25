module accel_file_player #(
//    parameter string FILE = "C:\Users\amand\CSE127A\sensorsoc\sensor_models\sensor_output\accel_digital.csv",
    parameter int FS_HZ = 50
)(
    input  logic clk,
    input  logic resetn,

    output logic sample_valid,
    output logic sample_ok,
    output logic signed [13:0] ax,
    output logic signed [13:0] ay,
    output logic signed [13:0] az
);

    int fd;
    int r;

    int divider = 50_000_000 / FS_HZ;
    int cnt;

    initial begin
        fd = $fopen("sim/data/accel_digital.csv", "r");
        if (fd == 0) begin
            $display("ERROR: Could not open accel CSV");
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

                r = $fscanf(fd, "%d,%d,%d\n", ax, ay, az);

                if (r == 3) begin
                    sample_valid <= 1;
                    sample_ok <= 1;
                end else begin
                    sample_valid <= 0;
                end
            end else begin
                sample_valid <= 0;
            end
        end
    end

endmodule
