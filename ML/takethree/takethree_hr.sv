module takethree_hr (
  input  logic              clk,
  input  logic              rstn,
  input  logic              in_valid,
  output logic              in_ready,
  input  logic signed [15:0] hr_q8,
  output logic              out_valid,
  input  logic              out_ready,
  output logic              state_out,
  output logic signed [31:0] logit0_q16,
  output logic signed [31:0] logit1_q16
);

  localparam int H1 = 16;

  // Use regular arrays + initial block (Icarus-compatible)
  logic signed [15:0] W1 [0:H1-1];
  logic signed [31:0] B1 [0:H1-1];
  logic signed [15:0] W2 [0:1][0:H1-1];
  logic signed [31:0] B2 [0:1];

  initial begin
    // W1
    W1[0]=-173;  W1[1]=79;    W1[2]=241;  W1[3]=-149;
    W1[4]=318;   W1[5]=-199;  W1[6]=123;  W1[7]=35;
    W1[8]=-26;   W1[9]=-212;  W1[10]=-204;W1[11]=-51;
    W1[12]=-95;  W1[13]=-62;  W1[14]=247; W1[15]=119;

    // B1
    B1[0]=-65121;  B1[1]=-51126;  B1[2]=60778;   B1[3]=-14126;
    B1[4]=15861;   B1[5]=29445;   B1[6]=56116;   B1[7]=-53580;
    B1[8]=54427;   B1[9]=-15848;  B1[10]=-11218; B1[11]=-52830;
    B1[12]=-56097; B1[13]=-49047; B1[14]=-37902; B1[15]=-69759;

    // W2 class 0
    W2[0][0]=-21;  W2[0][1]=492;  W2[0][2]=114;  W2[0][3]=-95;
    W2[0][4]=-116; W2[0][5]=15;   W2[0][6]=-12;  W2[0][7]=-44;
    W2[0][8]=36;   W2[0][9]=-236; W2[0][10]=-28; W2[0][11]=41;
    W2[0][12]=-63; W2[0][13]=-25; W2[0][14]=79;  W2[0][15]=626;

    // W2 class 1
    W2[1][0]=4;    W2[1][1]=-519; W2[1][2]=-125; W2[1][3]=76;
    W2[1][4]=75;   W2[1][5]=-20;  W2[1][6]=-57;  W2[1][7]=26;
    W2[1][8]=9;    W2[1][9]=252;  W2[1][10]=65;  W2[1][11]=-2;
    W2[1][12]=-1;  W2[1][13]=50;  W2[1][14]=14;  W2[1][15]=-624;

    // B2
    B2[0]=1163;
    B2[1]=9994;
  end

  // -----------------------------
  // FSM (fixed timing)
  // -----------------------------
  typedef enum logic [3:0] {
    S_IDLE,
    S_L1_CALC, S_L1_WRITE,
    S_L2_0_INIT, S_L2_0_MAC, S_L2_0_DONE,
    S_L2_1_INIT, S_L2_1_MAC, S_L2_1_DONE,
    S_DONE
  } st_t;

  st_t st;

  logic signed [15:0] x;
  logic signed [15:0] hidden [0:H1-1];  // Q8
  logic signed [31:0] acc;              // Q16 accumulator
  logic signed [31:0] logit0, logit1;
  logic signed [31:0] acc_next;

  int unsigned i;
  int unsigned j;

  assign in_ready  = (st == S_IDLE);
  assign out_valid = (st == S_DONE);

  function automatic logic signed [15:0] relu_q8(input logic signed [15:0] v);
    if (v < 0) relu_q8 = 16'sd0; else relu_q8 = v;
  endfunction

  always_comb begin
    acc_next = acc;
    // default: keep
    case (st)
      S_L1_CALC:     acc_next = B1[i] + ($signed(W1[i]) * $signed(x)); // Q16
      S_L2_0_MAC:    acc_next = acc + ($signed(W2[0][j]) * $signed(hidden[j])); // Q16
      S_L2_1_MAC:    acc_next = acc + ($signed(W2[1][j]) * $signed(hidden[j])); // Q16
      default:       acc_next = acc;
    endcase
  end

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      st <= S_IDLE;
      x <= '0;
      acc <= '0;
      logit0 <= '0;
      logit1 <= '0;
      logit0_q16 <= '0;
      logit1_q16 <= '0;
      state_out <= 1'b0;
      i <= 0; j <= 0;
      for (int k = 0; k < H1; k++) hidden[k] <= '0;
    end else begin
      case (st)
        S_IDLE: begin
          if (in_valid) begin
            x <= hr_q8;
            i <= 0;
            st <= S_L1_CALC;
          end
        end

        // Layer 1: one neuron per 2 cycles (calc then write)
        S_L1_CALC: begin
          acc <= acc_next;
          st  <= S_L1_WRITE;
        end

        S_L1_WRITE: begin
          // hidden_q8 = ReLU( acc_q16 >> 8 )
          hidden[i] <= relu_q8($signed(acc >>> 8));
          if (i == H1-1) begin
            st <= S_L2_0_INIT;
          end else begin
            i <= i + 1;
            st <= S_L1_CALC;
          end
        end

        // Layer 2 class 0
        S_L2_0_INIT: begin
          acc <= B2[0];
          j <= 0;
          st <= S_L2_0_MAC;
        end

        S_L2_0_MAC: begin
          acc <= acc_next;
          if (j == H1-1) st <= S_L2_0_DONE;
          else j <= j + 1;
        end

        S_L2_0_DONE: begin
          logit0 <= acc;
          st <= S_L2_1_INIT;
        end

        // Layer 2 class 1
        S_L2_1_INIT: begin
          acc <= B2[1];
          j <= 0;
          st <= S_L2_1_MAC;
        end

        S_L2_1_MAC: begin
          acc <= acc_next;
          if (j == H1-1) st <= S_L2_1_DONE;
          else j <= j + 1;
        end

        S_L2_1_DONE: begin
          logit1 <= acc;
          logit0_q16 <= logit0;
          logit1_q16 <= acc;
          state_out <= ($signed(acc) > $signed(logit0));
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
