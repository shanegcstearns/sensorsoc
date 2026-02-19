module takethree (
  input              clk,
  input              rstn,
  input              in_valid,
  output             in_ready,

  // ---- Quantized Q8 feature inputs ----
  // Order matches training:
  // movement, cosine, delta_hr, hr_rmssd
  input signed [15:0] movement_q8,
  input signed [15:0] cosine_q8,
  input signed [15:0] hr_q8,        // delta_hr  
  input signed [15:0] hr_rmssd_q8,

  output             out_valid,
  input              out_ready,

  output             state_out,
  output signed [31:0] logit0_q16,
  output signed [31:0] logit1_q16
);

  // --------------------------------------------------
  // Network size (fixed for iverilog stability)
  // --------------------------------------------------
  localparam integer H1 = 16;

  // --------------------------------------------------
  // Weight storage (filled by exporter)
  // --------------------------------------------------
  reg signed [15:0] W1 [0:15][0:3];
  reg signed [31:0] B1 [0:15];
  reg signed [15:0] W2 [0:1][0:15];
  reg signed [31:0] B2 [0:1];

  integer ii, jj;

  initial begin
    // Zero everything first (safe sim behavior)
    for (ii = 0; ii < 16; ii = ii + 1) begin
      B1[ii] = 0;
      for (jj = 0; jj < 4; jj = jj + 1)
        W1[ii][jj] = 0;
    end

    for (ii = 0; ii < 2; ii = ii + 1) begin
      B2[ii] = 0;
      for (jj = 0; jj < 16; jj = jj + 1)
        W2[ii][jj] = 0;
    end

    // --------------------------------------------------
    // PASTE EXPORTED WEIGHTS HERE
    // (from Python exporter)
    // --------------------------------------------------
  end

  // --------------------------------------------------
  // Handshake
  // --------------------------------------------------
  assign in_ready  = (st == S_IDLE);
  assign out_valid = (st == S_DONE);

  // --------------------------------------------------
  // Internal registers
  // --------------------------------------------------
  reg signed [15:0] x0, x1, x2, x3;     // movement, cosine, delta_hr, hr_rmssd
  reg signed [15:0] hidden [0:15];      // Q8
  reg signed [31:0] acc;                // Q16
  reg signed [31:0] logit0;

  integer i;
  integer j;

  // outputs
  reg state_r;
  reg signed [31:0] l0_r, l1_r;

  assign state_out  = state_r;
  assign logit0_q16 = l0_r;
  assign logit1_q16 = l1_r;

  // --------------------------------------------------
  // FSM States
  // --------------------------------------------------
  localparam [3:0]
    S_IDLE      = 4'd0,
    S_L1_INIT   = 4'd1,
    S_L1_MAC0   = 4'd2,
    S_L1_MAC1   = 4'd3,
    S_L1_MAC2   = 4'd4,
    S_L1_MAC3   = 4'd5,
    S_L1_WRITE  = 4'd6,
    S_L2_0_INIT = 4'd7,
    S_L2_0_MAC  = 4'd8,
    S_L2_0_DONE = 4'd9,
    S_L2_1_INIT = 4'd10,
    S_L2_1_MAC  = 4'd11,
    S_L2_1_DONE = 4'd12,
    S_DONE      = 4'd13;

  reg [3:0] st;

  // --------------------------------------------------
  // ReLU (Q8)
  // --------------------------------------------------
  function signed [15:0] relu_q8;
    input signed [15:0] v;
    begin
      if (v < 0)
        relu_q8 = 16'sd0;
      else
        relu_q8 = v;
    end
  endfunction

  // --------------------------------------------------
  // Sequential logic
  // --------------------------------------------------
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      st <= S_IDLE;
      acc <= 0;
      logit0 <= 0;
      state_r <= 0;
      l0_r <= 0;
      l1_r <= 0;

      x0 <= 0; x1 <= 0; x2 <= 0; x3 <= 0;

      i <= 0;
      j <= 0;

      for (ii = 0; ii < 16; ii = ii + 1)
        hidden[ii] <= 0;

    end else begin

      case (st)

        // ------------------------------------------
        // Idle
        // ------------------------------------------
        S_IDLE: begin
          if (in_valid) begin
            x0 <= movement_q8;
            x1 <= cosine_q8;
            x2 <= hr_q8;           // delta_hr
            x3 <= hr_rmssd_q8;

            i <= 0;
            st <= S_L1_INIT;
          end
        end

        // ------------------------------------------
        // Layer 1 neuron
        // ------------------------------------------
        S_L1_INIT: begin
          acc <= B1[i];
          st <= S_L1_MAC0;
        end

        S_L1_MAC0: begin
          acc <= acc + ($signed(W1[i][0]) * $signed(x0));
          st <= S_L1_MAC1;
        end

        S_L1_MAC1: begin
          acc <= acc + ($signed(W1[i][1]) * $signed(x1));
          st <= S_L1_MAC2;
        end

        S_L1_MAC2: begin
          acc <= acc + ($signed(W1[i][2]) * $signed(x2));
          st <= S_L1_MAC3;
        end

        S_L1_MAC3: begin
          acc <= acc + ($signed(W1[i][3]) * $signed(x3));
          st <= S_L1_WRITE;
        end

        S_L1_WRITE: begin
          hidden[i] <= relu_q8($signed(acc >>> 8));
          if (i == 15) begin
            j <= 0;
            st <= S_L2_0_INIT;
          end else begin
            i <= i + 1;
            st <= S_L1_INIT;
          end
        end

        // ------------------------------------------
        // Logit 0
        // ------------------------------------------
        S_L2_0_INIT: begin
          acc <= B2[0];
          j <= 0;
          st <= S_L2_0_MAC;
        end

        S_L2_0_MAC: begin
          acc <= acc + ($signed(W2[0][j]) * $signed(hidden[j]));
          if (j == 15)
            st <= S_L2_0_DONE;
          else
            j <= j + 1;
        end

        S_L2_0_DONE: begin
          logit0 <= acc;
          st <= S_L2_1_INIT;
        end

        // ------------------------------------------
        // Logit 1
        // ------------------------------------------
        S_L2_1_INIT: begin
          acc <= B2[1];
          j <= 0;
          st <= S_L2_1_MAC;
        end

        S_L2_1_MAC: begin
          acc <= acc + ($signed(W2[1][j]) * $signed(hidden[j]));
          if (j == 15)
            st <= S_L2_1_DONE;
          else
            j <= j + 1;
        end

        S_L2_1_DONE: begin
          l0_r <= logit0;
          l1_r <= acc;
          state_r <= ($signed(acc) > $signed(logit0));
          st <= S_DONE;
        end

        // ------------------------------------------
        // Done
        // ------------------------------------------
        S_DONE: begin
          if (out_ready)
            st <= S_IDLE;
        end

        default: st <= S_IDLE;

      endcase
    end
  end

endmodule
