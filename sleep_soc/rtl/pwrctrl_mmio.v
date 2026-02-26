`timescale 1ns/1ps

module pwrctrl_mmio #(
    parameter BASE_ADDR = 32'h0300_1000
)(
    input  wire        clk,
    input  wire        resetn,

    input  wire        mem_valid,
    input  wire [31:0] mem_addr,
    input  wire [31:0] mem_wdata,
    input  wire [3:0]  mem_wstrb,

    output reg         mem_ready,
    output reg  [31:0] mem_rdata,

    output reg         sleep_req_o,

    input  wire [31:0] wake_src_i,   // event bits from always-on blocks
    input  wire        cpu_awake_i    // cpu clock currently enabled
);

    localparam OFF_CTRL        = 32'h0;  // RW: bit0 sleep_req
    localparam OFF_WAKE_STATUS = 32'h4;  // R/W1C: latched wake flags
    localparam OFF_WAKE_REASON = 32'h8;  // RO: latched reason snapshot on wake

    wire sel = mem_valid && (mem_addr[31:12] == BASE_ADDR[31:12]); // 4KB page decode
    wire [31:0] off = mem_addr - BASE_ADDR;

    reg [31:0] wake_status;
    reg [31:0] wake_reason;

    reg cpu_awake_d;

    always @(posedge clk) begin
        if (!resetn) begin
            mem_ready   <= 1'b0;
            mem_rdata   <= 32'h0;
            sleep_req_o <= 1'b0;

            wake_status <= 32'h0;
            wake_reason <= 32'h0;

            cpu_awake_d <= 1'b1;
        end else begin
            mem_ready <= 1'b0;

            // ------------------------------------------------------------
            // 1) Continuously latch wake sources (sticky until W1C clears)
            // ------------------------------------------------------------
            wake_status <= wake_status | wake_src_i;

            // ------------------------------------------------------------
            // 2) Detect asleep->awake transition and snapshot wake reason
            // ------------------------------------------------------------
            cpu_awake_d <= cpu_awake_i;
            if (!cpu_awake_d && cpu_awake_i) begin
                wake_reason <= wake_status | wake_src_i;

                // IMPORTANT: clear sleep request on wake so we don't
                // immediately re-sleep in periodic scenarios.
                sleep_req_o <= 1'b0;
            end

            // ------------------------------------------------------------
            // 3) MMIO access
            // ------------------------------------------------------------
            if (sel) begin
                mem_ready <= 1'b1;

                // reads
                case (off)
                    OFF_CTRL:        mem_rdata <= {31'b0, sleep_req_o};
                    OFF_WAKE_STATUS: mem_rdata <= wake_status;
                    OFF_WAKE_REASON: mem_rdata <= wake_reason;
                    default:         mem_rdata <= 32'h0;
                endcase

                // writes
                if (mem_wstrb != 4'b0000) begin
                    case (off)
                        OFF_CTRL: begin
                            if (mem_wstrb[0])
                                sleep_req_o <= mem_wdata[0];
                        end

                        OFF_WAKE_STATUS: begin
                            // W1C: clear bits that are '1' in mem_wdata
                            // Note: include wake_src_i OR-in (same-cycle) before clear.
                            wake_status <= (wake_status | wake_src_i) & ~mem_wdata;
                        end

                        default: begin end
                    endcase
                end
            end
        end
    end

endmodule
