// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024ROOT_H_
#define VERILATED_VTOP___024ROOT_H_  // guard

#include "verilated.h"


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rstn,0,0);
    VL_IN8(in_valid,0,0);
    VL_OUT8(in_ready,0,0);
    VL_OUT8(out_valid,0,0);
    VL_IN8(out_ready,0,0);
    VL_OUT8(state_out,0,0);
    CData/*0:0*/ takethree__DOT__clk;
    CData/*0:0*/ takethree__DOT__rstn;
    CData/*0:0*/ takethree__DOT__in_valid;
    CData/*0:0*/ takethree__DOT__in_ready;
    CData/*0:0*/ takethree__DOT__out_valid;
    CData/*0:0*/ takethree__DOT__out_ready;
    CData/*0:0*/ takethree__DOT__state_out;
    CData/*0:0*/ takethree__DOT__state_r;
    CData/*3:0*/ takethree__DOT__st;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__takethree__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__takethree__DOT__rstn__0;
    VL_IN16(movement_q8,15,0);
    VL_IN16(cosine_q8,15,0);
    VL_IN16(hr_q8,15,0);
    VL_IN16(hr_rmssd_q8,15,0);
    SData/*15:0*/ takethree__DOT__movement_q8;
    SData/*15:0*/ takethree__DOT__cosine_q8;
    SData/*15:0*/ takethree__DOT__hr_q8;
    SData/*15:0*/ takethree__DOT__hr_rmssd_q8;
    SData/*15:0*/ takethree__DOT__x0;
    SData/*15:0*/ takethree__DOT__x1;
    SData/*15:0*/ takethree__DOT__x2;
    SData/*15:0*/ takethree__DOT__x3;
    VL_OUT(logit0_q16,31,0);
    VL_OUT(logit1_q16,31,0);
    IData/*31:0*/ takethree__DOT__logit0_q16;
    IData/*31:0*/ takethree__DOT__logit1_q16;
    IData/*31:0*/ takethree__DOT__ii;
    IData/*31:0*/ takethree__DOT__jj;
    IData/*31:0*/ takethree__DOT__acc;
    IData/*31:0*/ takethree__DOT__logit0;
    IData/*31:0*/ takethree__DOT__l0_r;
    IData/*31:0*/ takethree__DOT__l1_r;
    IData/*31:0*/ takethree__DOT__i;
    IData/*31:0*/ takethree__DOT__j;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<VlUnpacked<SData/*15:0*/, 4>, 16> takethree__DOT__W1;
    VlUnpacked<IData/*31:0*/, 16> takethree__DOT__B1;
    VlUnpacked<VlUnpacked<SData/*15:0*/, 16>, 2> takethree__DOT__W2;
    VlUnpacked<IData/*31:0*/, 2> takethree__DOT__B2;
    VlUnpacked<SData/*15:0*/, 16> takethree__DOT__hidden;
    VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VicoTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    VlNBACommitQueue<VlUnpacked<SData/*15:0*/, 16>, false, SData/*15:0*/, 1> __VdlyCommitQueuetakethree__DOT__hidden;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr CData/*3:0*/ takethree__DOT__S_IDLE = 0U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L1_INIT = 1U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L1_MAC0 = 2U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L1_MAC1 = 3U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L1_MAC2 = 4U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L1_MAC3 = 5U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L1_WRITE = 6U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L2_0_INIT = 7U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L2_0_MAC = 8U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L2_0_DONE = 9U;
    static constexpr CData/*3:0*/ takethree__DOT__S_L2_1_INIT = 0x0aU;
    static constexpr CData/*3:0*/ takethree__DOT__S_L2_1_MAC = 0x0bU;
    static constexpr CData/*3:0*/ takethree__DOT__S_L2_1_DONE = 0x0cU;
    static constexpr CData/*3:0*/ takethree__DOT__S_DONE = 0x0dU;
    static constexpr IData/*31:0*/ takethree__DOT__H1 = 0x00000010U;
    static constexpr IData/*31:0*/ takethree__DOT__SCALE_Q0 = 0x000005a4U;
    static constexpr IData/*31:0*/ takethree__DOT__SCALE_Q1 = 0x00007801U;
    static constexpr IData/*31:0*/ takethree__DOT__SCALE_Q2 = 0x00003297U;
    static constexpr IData/*31:0*/ takethree__DOT__SCALE_Q3 = 0x000f233aU;

    // CONSTRUCTORS
    Vtop___024root(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024root();
    VL_UNCOPYABLE(Vtop___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
