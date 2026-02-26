`timescale 1ns/1ps

module timer_mmio #(
    parameter BASE_ADDR = 32'h0300_2000
)(
    input  wire        clk,
    input  wire        resetn,

    input  wire        mem_valid,
    input  wire [31:0] mem_addr,
    input  wire [31:0] mem_wdata,
    input  wire [3:0]  mem_wstrb,

    output reg         mem_ready,
    output reg  [31:0] mem_rdata,

    output wire        event_o,
    output reg  [31:0] rdata_o
);

    localparam [31:0] OFF_CTRL   = 32'h0;  // bit0 enable, bit1 periodic
    localparam [31:0] OFF_RELOAD = 32'h4;
    localparam [31:0] OFF_COUNT  = 32'h8;
    localparam [31:0] OFF_EVENT  = 32'hC;  // W1C bit0, read shows latched

    // 4KB page decode (matches your other MMIOs)
    wire        sel = mem_valid && (mem_addr[31:12] == BASE_ADDR[31:12]);
    wire [31:0] off = mem_addr - BASE_ADDR;

    reg         enable, periodic;
    reg [31:0]  reload, count;
    reg         event_latched;

    assign event_o = event_latched;

    // Icarus-friendly "read mux" as a reg set inside sequential block
    reg [31:0] read_data;

    wire wr = sel && (mem_wstrb != 4'b0000);

    always @(posedge clk) begin
        if (!resetn) begin
            // timer regs
            enable        <= 1'b0;
            periodic      <= 1'b0;
            reload        <= 32'd5_000_000;
            count         <= 32'd5_000_000;
            event_latched <= 1'b0;

            // bus regs
            mem_ready <= 1'b0;
            mem_rdata <= 32'h0;
            rdata_o   <= 32'h0;

            read_data <= 32'h0;
        end else begin
            // default
            mem_ready <= 1'b0;

            // -------------------------
            // Timer countdown (always-on)
            // -------------------------
            if (enable) begin
                if (count != 32'd0) begin
                    count <= count - 32'd1;
                end else begin
                    event_latched <= 1'b1;
                    if (periodic) begin
                        count <= reload;
                    end else begin
                        enable <= 1'b0;
                    end
                end
            end

            // -------------------------
            // MMIO access (1-cycle ready)
            // -------------------------
            if (sel) begin
                mem_ready <= 1'b1;

                // Build read_data *fresh* (no stale rdata issues)
                read_data = 32'h0;
                case (off)
                    OFF_CTRL:   read_data = {30'b0, periodic, enable};
                    OFF_RELOAD: read_data = reload;
                    OFF_COUNT:  read_data = count;
                    OFF_EVENT:  read_data = {31'b0, event_latched};
                    default:    read_data = 32'h0;
                endcase

                // Drive outputs from the same computed value
                mem_rdata <= read_data;
                rdata_o   <= read_data;

                // Writes
                if (wr) begin
                    case (off)
                        OFF_CTRL: begin
                            // Your bus is 32-bit, but typically only byte0 matters.
                            // Respect wstrb[0] so firmware can byte-write.
                            if (mem_wstrb[0]) begin
                                enable   <= mem_wdata[0];
                                periodic <= mem_wdata[1];
                            end
                        end

                        OFF_RELOAD: begin
                            reload <= mem_wdata;
                        end

                        OFF_COUNT: begin
                            count <= mem_wdata;
                        end

                        OFF_EVENT: begin
                            // W1C bit0
                            if (mem_wdata[0]) event_latched <= 1'b0;
                        end
                        default: begin end
                    endcase
                end
            end
        end
    end

endmodule
