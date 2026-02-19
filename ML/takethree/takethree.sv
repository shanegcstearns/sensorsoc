// Auto-generated: 4-feature MLP -> Icarus-safe SystemVerilog
// Features (training order): movement, cosine, delta_hr, hr_rmssd
// Port names:
//   movement_q8  -> movement
//   cosine_q8    -> cosine
//   hr_q8        -> delta_hr   (not raw hr!)
//   hr_rmssd_q8  -> hr_rmssd
// Runtime quantization (done in TB/SW):
//   x_q8[i] = round(x_float[i] * SCALE_Q[i])
// Math:
//   acc_q16 = b_q16 + sum(w_q8 * x_q8)
//   hidden_q8 = ReLU(acc_q16 >>> 8)
//   logits_q16 similarly; state = (logit1 > logit0)

module takethree (
  input              clk,
  input              rstn,
  input              in_valid,
  output             in_ready,
  input signed [15:0] movement_q8,
  input signed [15:0] cosine_q8,
  input signed [15:0] hr_q8,
  input signed [15:0] hr_rmssd_q8,
  output             out_valid,
  input              out_ready,
  output             state_out,
  output signed [31:0] logit0_q16,
  output signed [31:0] logit1_q16
);

  localparam integer H1 = 16;

  // Per-feature integer scales used by TB/SW
  // x_q8 = round(x_float * SCALE_Q)
  localparam integer SCALE_Q0 = 1444; // movement
  localparam integer SCALE_Q1 = 30721; // cosine
  localparam integer SCALE_Q2 = 12951; // delta_hr
  localparam integer SCALE_Q3 = 992058; // hr_rmssd

  // Fixed-size storage (Icarus-friendly)
  logic signed [15:0] W1 [0:15][0:3];
  logic signed [31:0] B1 [0:15];
  logic signed [15:0] W2 [0:1][0:15];
  logic signed [31:0] B2 [0:1];

  integer ii;
  integer jj;
  initial begin
    B1[0] = 32'sd336;
    W1[0][0] = -16'sd59;
    W1[0][1] = 16'sd32;
    W1[0][2] = 16'sd101;
    W1[0][3] = -16'sd54;
    B1[1] = -32'sd11100;
    W1[1][0] = 16'sd129;
    W1[1][1] = -16'sd108;
    W1[1][2] = 16'sd74;
    W1[1][3] = 16'sd56;
    B1[2] = -32'sd29304;
    W1[2][0] = -16'sd5;
    W1[2][1] = -16'sd20;
    W1[2][2] = -16'sd59;
    W1[2][3] = -16'sd7;
    B1[3] = -32'sd31901;
    W1[3][0] = -16'sd8;
    W1[3][1] = -16'sd70;
    W1[3][2] = 16'sd68;
    W1[3][3] = 16'sd64;
    B1[4] = -32'sd14973;
    W1[4][0] = -16'sd70;
    W1[4][1] = -16'sd92;
    W1[4][2] = 16'sd78;
    W1[4][3] = 16'sd48;
    B1[5] = 32'sd53847;
    W1[5][0] = -16'sd42;
    W1[5][1] = 16'sd60;
    W1[5][2] = 16'sd120;
    W1[5][3] = 16'sd6;
    B1[6] = -32'sd2216;
    W1[6][0] = 16'sd153;
    W1[6][1] = -16'sd31;
    W1[6][2] = 16'sd24;
    W1[6][3] = -16'sd6;
    B1[7] = -32'sd38996;
    W1[7][0] = -16'sd70;
    W1[7][1] = -16'sd76;
    W1[7][2] = -16'sd54;
    W1[7][3] = -16'sd101;
    B1[8] = -32'sd477;
    W1[8][0] = -16'sd11;
    W1[8][1] = 16'sd29;
    W1[8][2] = 16'sd85;
    W1[8][3] = 16'sd16;
    B1[9] = -32'sd16506;
    W1[9][0] = -16'sd114;
    W1[9][1] = 16'sd99;
    W1[9][2] = -16'sd29;
    W1[9][3] = -16'sd126;
    B1[10] = 32'sd37305;
    W1[10][0] = 16'sd89;
    W1[10][1] = -16'sd122;
    W1[10][2] = 16'sd64;
    W1[10][3] = 16'sd45;
    B1[11] = 32'sd42818;
    W1[11][0] = -16'sd77;
    W1[11][1] = -16'sd87;
    W1[11][2] = 16'sd40;
    W1[11][3] = -16'sd12;
    B1[12] = -32'sd28532;
    W1[12][0] = 16'sd18;
    W1[12][1] = -16'sd44;
    W1[12][2] = -16'sd72;
    W1[12][3] = -16'sd83;
    B1[13] = -32'sd13734;
    W1[13][0] = 16'sd42;
    W1[13][1] = -16'sd92;
    W1[13][2] = -16'sd83;
    W1[13][3] = 16'sd105;
    B1[14] = 32'sd28745;
    W1[14][0] = -16'sd25;
    W1[14][1] = 16'sd126;
    W1[14][2] = -16'sd37;
    W1[14][3] = -16'sd4;
    B1[15] = 32'sd34781;
    W1[15][0] = -16'sd9;
    W1[15][1] = 16'sd91;
    W1[15][2] = 16'sd88;
    W1[15][3] = 16'sd98;
    B2[0] = -32'sd27521;
    B2[1] = 32'sd18260;
    W2[0][0] = -16'sd18;
    W2[0][1] = 16'sd18;
    W2[0][2] = 16'sd17;
    W2[0][3] = 16'sd24;
    W2[0][4] = 16'sd41;
    W2[0][5] = -16'sd24;
    W2[0][6] = 16'sd62;
    W2[0][7] = 16'sd15;
    W2[0][8] = -16'sd14;
    W2[0][9] = -16'sd45;
    W2[0][10] = -16'sd7;
    W2[0][11] = -16'sd31;
    W2[0][12] = 16'sd9;
    W2[0][13] = -16'sd13;
    W2[0][14] = 16'sd0;
    W2[0][15] = 16'sd18;
    W2[1][0] = -16'sd43;
    W2[1][1] = 16'sd6;
    W2[1][2] = -16'sd7;
    W2[1][3] = -16'sd28;
    W2[1][4] = -16'sd42;
    W2[1][5] = -16'sd8;
    W2[1][6] = 16'sd25;
    W2[1][7] = -16'sd2;
    W2[1][8] = 16'sd28;
    W2[1][9] = -16'sd37;
    W2[1][10] = 16'sd62;
    W2[1][11] = 16'sd53;
    W2[1][12] = -16'sd22;
    W2[1][13] = -16'sd3;
    W2[1][14] = 16'sd9;
    W2[1][15] = 16'sd36;
  end

  assign in_ready  = (st == S_IDLE);
  assign out_valid = (st == S_DONE);

  logic signed [15:0] x0, x1, x2, x3;
  logic signed [15:0] hidden [0:15];
  logic signed [31:0] acc;
  logic signed [31:0] logit0;
  logic signed [31:0] l0_r, l1_r;
  logic state_r;
  assign logit0_q16 = l0_r;
  assign logit1_q16 = l1_r;
  assign state_out  = state_r;

  integer i;
  integer j;

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

  logic [3:0] st;

  function signed [15:0] relu_q8;
    input signed [15:0] v;
    begin
      if (v < 0) relu_q8 = 16'sd0; else relu_q8 = v;
    end
  endfunction

  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      st <= S_IDLE;
      acc <= 0;
      logit0 <= 0;
      l0_r <= 0;
      l1_r <= 0;
      state_r <= 0;
      i <= 0;
      j <= 0;
      x0 <= 0; x1 <= 0; x2 <= 0; x3 <= 0;
      for (ii = 0; ii < 16; ii = ii + 1) hidden[ii] <= 0;
    end else begin
      case (st)

        S_IDLE: begin
          if (in_valid) begin
            x0 <= movement_q8;
            x1 <= cosine_q8;
            x2 <= hr_q8;         // delta_hr
            x3 <= hr_rmssd_q8;
            i <= 0;
            st <= S_L1_INIT;
          end
        end

        S_L1_INIT: begin
          acc <= B1[i];
          st <= S_L1_MAC0;
        end
        S_L1_MAC0: begin acc <= acc + ($signed(W1[i][0]) * $signed(x0)); st <= S_L1_MAC1; end
        S_L1_MAC1: begin acc <= acc + ($signed(W1[i][1]) * $signed(x1)); st <= S_L1_MAC2; end
        S_L1_MAC2: begin acc <= acc + ($signed(W1[i][2]) * $signed(x2)); st <= S_L1_MAC3; end
        S_L1_MAC3: begin acc <= acc + ($signed(W1[i][3]) * $signed(x3)); st <= S_L1_WRITE; end
        S_L1_WRITE: begin
          hidden[i] <= relu_q8($signed(acc >>> 8)[15:0]);
          if (i == 15) begin
            j <= 0;
            st <= S_L2_0_INIT;
          end else begin
            i <= i + 1;
            st <= S_L1_INIT;
          end
        end

        S_L2_0_INIT: begin
          acc <= B2[0];
          j <= 0;
          st <= S_L2_0_MAC;
        end
        S_L2_0_MAC: begin
          acc <= acc + ($signed(W2[0][j]) * $signed(hidden[j]));
          if (j == 15) st <= S_L2_0_DONE;
          else j <= j + 1;
        end
        S_L2_0_DONE: begin
          logit0 <= acc;
          st <= S_L2_1_INIT;
        end

        S_L2_1_INIT: begin
          acc <= B2[1];
          j <= 0;
          st <= S_L2_1_MAC;
        end
        S_L2_1_MAC: begin
          acc <= acc + ($signed(W2[1][j]) * $signed(hidden[j]));
          if (j == 15) st <= S_L2_1_DONE;
          else j <= j + 1;
        end
        S_L2_1_DONE: begin
          l0_r <= logit0;
          l1_r <= acc;
          state_r <= ($signed(acc) > $signed(logit0));
          st <= S_DONE;
        end

        S_DONE: begin
          if (out_ready) st <= S_IDLE;
        end

        default: st <= S_IDLE;
      endcase
    end
  end

endmodule
