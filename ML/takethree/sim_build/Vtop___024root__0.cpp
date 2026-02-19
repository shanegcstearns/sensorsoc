// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__ico\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VicoTriggered
                                      [0U]) | (IData)((IData)(vlSelfRef.__VicoFirstIteration)));
    vlSelfRef.__VicoFirstIteration = 0U;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
    }
#endif
}

bool Vtop___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___trigger_anySet__ico\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.takethree__DOT__clk = vlSelfRef.clk;
    vlSelfRef.takethree__DOT__rstn = vlSelfRef.rstn;
    vlSelfRef.takethree__DOT__in_valid = vlSelfRef.in_valid;
    vlSelfRef.takethree__DOT__movement_q8 = vlSelfRef.movement_q8;
    vlSelfRef.takethree__DOT__cosine_q8 = vlSelfRef.cosine_q8;
    vlSelfRef.takethree__DOT__hr_q8 = vlSelfRef.hr_q8;
    vlSelfRef.takethree__DOT__hr_rmssd_q8 = vlSelfRef.hr_rmssd_q8;
    vlSelfRef.takethree__DOT__out_ready = vlSelfRef.out_ready;
    vlSelfRef.takethree__DOT__in_ready = (0U == (IData)(vlSelfRef.takethree__DOT__st));
    vlSelfRef.takethree__DOT__out_valid = (0x0dU == (IData)(vlSelfRef.takethree__DOT__st));
    vlSelfRef.takethree__DOT__state_out = vlSelfRef.takethree__DOT__state_r;
    vlSelfRef.takethree__DOT__logit0_q16 = vlSelfRef.takethree__DOT__l0_r;
    vlSelfRef.takethree__DOT__logit1_q16 = vlSelfRef.takethree__DOT__l1_r;
    vlSelfRef.in_ready = vlSelfRef.takethree__DOT__in_ready;
    vlSelfRef.out_valid = vlSelfRef.takethree__DOT__out_valid;
    vlSelfRef.state_out = vlSelfRef.takethree__DOT__state_out;
    vlSelfRef.logit0_q16 = vlSelfRef.takethree__DOT__logit0_q16;
    vlSelfRef.logit1_q16 = vlSelfRef.takethree__DOT__logit1_q16;
}

void Vtop___024root___eval_ico(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_ico\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VicoTriggered[0U])) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

bool Vtop___024root___eval_phase__ico(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__ico\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VicoExecute;
    // Body
    Vtop___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = Vtop___024root___trigger_anySet__ico(vlSelfRef.__VicoTriggered);
    if (__VicoExecute) {
        Vtop___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__act\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                    ((((~ (IData)(vlSelfRef.takethree__DOT__rstn)) 
                                                       & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__takethree__DOT__rstn__0)) 
                                                      << 1U) 
                                                     | ((IData)(vlSelfRef.takethree__DOT__clk) 
                                                        & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__takethree__DOT__clk__0))))));
    vlSelfRef.__Vtrigprevexpr___TOP__takethree__DOT__clk__0 
        = vlSelfRef.takethree__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__takethree__DOT__rstn__0 
        = vlSelfRef.takethree__DOT__rstn;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
}

bool Vtop___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___trigger_anySet__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void Vtop___024root___nba_sequent__TOP__0(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__0\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    SData/*15:0*/ __Vfunc_takethree__DOT__relu_q8__0__Vfuncout;
    __Vfunc_takethree__DOT__relu_q8__0__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_takethree__DOT__relu_q8__0__v;
    __Vfunc_takethree__DOT__relu_q8__0__v = 0;
    CData/*3:0*/ __Vdly__takethree__DOT__st;
    __Vdly__takethree__DOT__st = 0;
    IData/*31:0*/ __Vdly__takethree__DOT__acc;
    __Vdly__takethree__DOT__acc = 0;
    IData/*31:0*/ __Vdly__takethree__DOT__j;
    __Vdly__takethree__DOT__j = 0;
    IData/*31:0*/ __Vdly__takethree__DOT__i;
    __Vdly__takethree__DOT__i = 0;
    SData/*15:0*/ __VdlyVal__takethree__DOT__hidden__v0;
    __VdlyVal__takethree__DOT__hidden__v0 = 0;
    CData/*3:0*/ __VdlyDim0__takethree__DOT__hidden__v0;
    __VdlyDim0__takethree__DOT__hidden__v0 = 0;
    CData/*3:0*/ __VdlyDim0__takethree__DOT__hidden__v1;
    __VdlyDim0__takethree__DOT__hidden__v1 = 0;
    // Body
    __Vdly__takethree__DOT__acc = vlSelfRef.takethree__DOT__acc;
    __Vdly__takethree__DOT__j = vlSelfRef.takethree__DOT__j;
    __Vdly__takethree__DOT__i = vlSelfRef.takethree__DOT__i;
    __Vdly__takethree__DOT__st = vlSelfRef.takethree__DOT__st;
    if (vlSelfRef.takethree__DOT__rstn) {
        if ((8U & (IData)(vlSelfRef.takethree__DOT__st))) {
            if ((4U & (IData)(vlSelfRef.takethree__DOT__st))) {
                if ((2U & (IData)(vlSelfRef.takethree__DOT__st))) {
                    __Vdly__takethree__DOT__st = 0U;
                } else if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
                    if (vlSelfRef.takethree__DOT__out_ready) {
                        __Vdly__takethree__DOT__st = 0U;
                    }
                } else {
                    vlSelfRef.takethree__DOT__l0_r 
                        = vlSelfRef.takethree__DOT__logit0;
                    vlSelfRef.takethree__DOT__l1_r 
                        = vlSelfRef.takethree__DOT__acc;
                    vlSelfRef.takethree__DOT__state_r 
                        = VL_GTS_III(32, vlSelfRef.takethree__DOT__acc, vlSelfRef.takethree__DOT__logit0);
                    __Vdly__takethree__DOT__st = 0x0dU;
                }
            } else if ((2U & (IData)(vlSelfRef.takethree__DOT__st))) {
                if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
                    __Vdly__takethree__DOT__acc = (vlSelfRef.takethree__DOT__acc 
                                                   + 
                                                   VL_MULS_III(32, 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelfRef.takethree__DOT__W2
                                                                             [1U]
                                                                             [
                                                                             (0x0000000fU 
                                                                              & vlSelfRef.takethree__DOT__j)]), 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelfRef.takethree__DOT__hidden
                                                                             [
                                                                             (0x0000000fU 
                                                                              & vlSelfRef.takethree__DOT__j)])));
                    if ((0x0000000fU == vlSelfRef.takethree__DOT__j)) {
                        __Vdly__takethree__DOT__st = 0x0cU;
                    } else {
                        __Vdly__takethree__DOT__j = 
                            ((IData)(1U) + vlSelfRef.takethree__DOT__j);
                    }
                } else {
                    __Vdly__takethree__DOT__acc = vlSelfRef.takethree__DOT__B2
                        [1U];
                    __Vdly__takethree__DOT__j = 0U;
                    __Vdly__takethree__DOT__st = 0x0bU;
                }
            } else if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
                vlSelfRef.takethree__DOT__logit0 = vlSelfRef.takethree__DOT__acc;
                __Vdly__takethree__DOT__st = 0x0aU;
            } else {
                __Vdly__takethree__DOT__acc = (vlSelfRef.takethree__DOT__acc 
                                               + VL_MULS_III(32, 
                                                             VL_EXTENDS_II(32,16, 
                                                                           vlSelfRef.takethree__DOT__W2
                                                                           [0U]
                                                                           [
                                                                           (0x0000000fU 
                                                                            & vlSelfRef.takethree__DOT__j)]), 
                                                             VL_EXTENDS_II(32,16, 
                                                                           vlSelfRef.takethree__DOT__hidden
                                                                           [
                                                                           (0x0000000fU 
                                                                            & vlSelfRef.takethree__DOT__j)])));
                if ((0x0000000fU == vlSelfRef.takethree__DOT__j)) {
                    __Vdly__takethree__DOT__st = 9U;
                } else {
                    __Vdly__takethree__DOT__j = ((IData)(1U) 
                                                 + vlSelfRef.takethree__DOT__j);
                }
            }
        } else if ((4U & (IData)(vlSelfRef.takethree__DOT__st))) {
            if ((2U & (IData)(vlSelfRef.takethree__DOT__st))) {
                if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
                    __Vdly__takethree__DOT__acc = vlSelfRef.takethree__DOT__B2
                        [0U];
                    __Vdly__takethree__DOT__j = 0U;
                    __Vdly__takethree__DOT__st = 8U;
                } else {
                    __Vfunc_takethree__DOT__relu_q8__0__v 
                        = (0x0000ffffU & VL_SHIFTRS_III(16,32,32, vlSelfRef.takethree__DOT__acc, 8U));
                    __Vfunc_takethree__DOT__relu_q8__0__Vfuncout 
                        = (VL_GTS_III(32, 0U, VL_EXTENDS_II(32,16, (IData)(__Vfunc_takethree__DOT__relu_q8__0__v)))
                            ? 0U : (IData)(__Vfunc_takethree__DOT__relu_q8__0__v));
                    __VdlyVal__takethree__DOT__hidden__v0 
                        = __Vfunc_takethree__DOT__relu_q8__0__Vfuncout;
                    __VdlyDim0__takethree__DOT__hidden__v0 
                        = (0x0000000fU & vlSelfRef.takethree__DOT__i);
                    vlSelfRef.__VdlyCommitQueuetakethree__DOT__hidden.enqueue(__VdlyVal__takethree__DOT__hidden__v0, (IData)(__VdlyDim0__takethree__DOT__hidden__v0));
                    if ((0x0000000fU == vlSelfRef.takethree__DOT__i)) {
                        __Vdly__takethree__DOT__j = 0U;
                        __Vdly__takethree__DOT__st = 7U;
                    } else {
                        __Vdly__takethree__DOT__i = 
                            ((IData)(1U) + vlSelfRef.takethree__DOT__i);
                        __Vdly__takethree__DOT__st = 1U;
                    }
                }
            } else if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
                __Vdly__takethree__DOT__acc = (vlSelfRef.takethree__DOT__acc 
                                               + VL_MULS_III(32, 
                                                             VL_EXTENDS_II(32,16, 
                                                                           vlSelfRef.takethree__DOT__W1
                                                                           [
                                                                           (0x0000000fU 
                                                                            & vlSelfRef.takethree__DOT__i)]
                                                                           [3U]), 
                                                             VL_EXTENDS_II(32,16, (IData)(vlSelfRef.takethree__DOT__x3))));
                __Vdly__takethree__DOT__st = 6U;
            } else {
                __Vdly__takethree__DOT__acc = (vlSelfRef.takethree__DOT__acc 
                                               + VL_MULS_III(32, 
                                                             VL_EXTENDS_II(32,16, 
                                                                           vlSelfRef.takethree__DOT__W1
                                                                           [
                                                                           (0x0000000fU 
                                                                            & vlSelfRef.takethree__DOT__i)]
                                                                           [2U]), 
                                                             VL_EXTENDS_II(32,16, (IData)(vlSelfRef.takethree__DOT__x2))));
                __Vdly__takethree__DOT__st = 5U;
            }
        } else if ((2U & (IData)(vlSelfRef.takethree__DOT__st))) {
            if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
                __Vdly__takethree__DOT__acc = (vlSelfRef.takethree__DOT__acc 
                                               + VL_MULS_III(32, 
                                                             VL_EXTENDS_II(32,16, 
                                                                           vlSelfRef.takethree__DOT__W1
                                                                           [
                                                                           (0x0000000fU 
                                                                            & vlSelfRef.takethree__DOT__i)]
                                                                           [1U]), 
                                                             VL_EXTENDS_II(32,16, (IData)(vlSelfRef.takethree__DOT__x1))));
                __Vdly__takethree__DOT__st = 4U;
            } else {
                __Vdly__takethree__DOT__acc = (vlSelfRef.takethree__DOT__acc 
                                               + VL_MULS_III(32, 
                                                             VL_EXTENDS_II(32,16, 
                                                                           vlSelfRef.takethree__DOT__W1
                                                                           [
                                                                           (0x0000000fU 
                                                                            & vlSelfRef.takethree__DOT__i)]
                                                                           [0U]), 
                                                             VL_EXTENDS_II(32,16, (IData)(vlSelfRef.takethree__DOT__x0))));
                __Vdly__takethree__DOT__st = 3U;
            }
        } else if ((1U & (IData)(vlSelfRef.takethree__DOT__st))) {
            __Vdly__takethree__DOT__acc = vlSelfRef.takethree__DOT__B1
                [(0x0000000fU & vlSelfRef.takethree__DOT__i)];
            __Vdly__takethree__DOT__st = 2U;
        } else if (vlSelfRef.takethree__DOT__in_valid) {
            __Vdly__takethree__DOT__i = 0U;
            vlSelfRef.takethree__DOT__x0 = vlSelfRef.takethree__DOT__movement_q8;
            vlSelfRef.takethree__DOT__x1 = vlSelfRef.takethree__DOT__cosine_q8;
            vlSelfRef.takethree__DOT__x2 = vlSelfRef.takethree__DOT__hr_q8;
            vlSelfRef.takethree__DOT__x3 = vlSelfRef.takethree__DOT__hr_rmssd_q8;
            __Vdly__takethree__DOT__st = 1U;
        }
    } else {
        __Vdly__takethree__DOT__i = 0U;
        vlSelfRef.takethree__DOT__ii = 0U;
        __Vdly__takethree__DOT__st = 0U;
        __Vdly__takethree__DOT__acc = 0U;
        vlSelfRef.takethree__DOT__logit0 = 0U;
        vlSelfRef.takethree__DOT__l0_r = 0U;
        vlSelfRef.takethree__DOT__l1_r = 0U;
        vlSelfRef.takethree__DOT__state_r = 0U;
        __Vdly__takethree__DOT__j = 0U;
        vlSelfRef.takethree__DOT__x0 = 0U;
        vlSelfRef.takethree__DOT__x1 = 0U;
        vlSelfRef.takethree__DOT__x2 = 0U;
        vlSelfRef.takethree__DOT__x3 = 0U;
        while (VL_GTS_III(32, 0x00000010U, vlSelfRef.takethree__DOT__ii)) {
            __VdlyDim0__takethree__DOT__hidden__v1 
                = (0x0000000fU & vlSelfRef.takethree__DOT__ii);
            vlSelfRef.__VdlyCommitQueuetakethree__DOT__hidden.enqueue(0U, (IData)(__VdlyDim0__takethree__DOT__hidden__v1));
            vlSelfRef.takethree__DOT__ii = ((IData)(1U) 
                                            + vlSelfRef.takethree__DOT__ii);
        }
    }
    vlSelfRef.takethree__DOT__acc = __Vdly__takethree__DOT__acc;
    vlSelfRef.takethree__DOT__j = __Vdly__takethree__DOT__j;
    vlSelfRef.__VdlyCommitQueuetakethree__DOT__hidden.commit(vlSelfRef.takethree__DOT__hidden);
    vlSelfRef.takethree__DOT__i = __Vdly__takethree__DOT__i;
    vlSelfRef.takethree__DOT__st = __Vdly__takethree__DOT__st;
    vlSelfRef.takethree__DOT__logit0_q16 = vlSelfRef.takethree__DOT__l0_r;
    vlSelfRef.takethree__DOT__logit1_q16 = vlSelfRef.takethree__DOT__l1_r;
    vlSelfRef.takethree__DOT__state_out = vlSelfRef.takethree__DOT__state_r;
    vlSelfRef.takethree__DOT__in_ready = (0U == (IData)(vlSelfRef.takethree__DOT__st));
    vlSelfRef.takethree__DOT__out_valid = (0x0dU == (IData)(vlSelfRef.takethree__DOT__st));
    vlSelfRef.logit0_q16 = vlSelfRef.takethree__DOT__logit0_q16;
    vlSelfRef.logit1_q16 = vlSelfRef.takethree__DOT__logit1_q16;
    vlSelfRef.state_out = vlSelfRef.takethree__DOT__state_out;
    vlSelfRef.in_ready = vlSelfRef.takethree__DOT__in_ready;
    vlSelfRef.out_valid = vlSelfRef.takethree__DOT__out_valid;
}

void Vtop___024root___eval_nba(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_nba\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vtop___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___trigger_orInto__act(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___trigger_orInto__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = (out[n] | in[n]);
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtop___024root___eval_phase__act(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__act\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtop___024root___eval_triggers__act(vlSelf);
    Vtop___024root___trigger_orInto__act(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    return (0U);
}

void Vtop___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtop___024root___eval_phase__nba(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__nba\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtop___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vtop___024root___eval_nba(vlSelf);
        Vtop___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vtop___024root___eval(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VicoIterCount;
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VicoIterCount = 0U;
    vlSelfRef.__VicoFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00000064U < __VicoIterCount)))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
#endif
            VL_FATAL_MT("/home/jfriday/sensorsoc/ML/takethree/takethree.sv", 15, "", "Input combinational region did not converge after 100 tries");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
    } while (Vtop___024root___eval_phase__ico(vlSelf));
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00000064U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("/home/jfriday/sensorsoc/ML/takethree/takethree.sv", 15, "", "NBA region did not converge after 100 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00000064U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                VL_FATAL_MT("/home/jfriday/sensorsoc/ML/takethree/takethree.sv", 15, "", "Active region did not converge after 100 tries");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
        } while (Vtop___024root___eval_phase__act(vlSelf));
    } while (Vtop___024root___eval_phase__nba(vlSelf));
}

#ifdef VL_DEBUG
void Vtop___024root___eval_debug_assertions(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_debug_assertions\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");
    }
    if (VL_UNLIKELY(((vlSelfRef.rstn & 0xfeU)))) {
        Verilated::overWidthError("rstn");
    }
    if (VL_UNLIKELY(((vlSelfRef.in_valid & 0xfeU)))) {
        Verilated::overWidthError("in_valid");
    }
    if (VL_UNLIKELY(((vlSelfRef.out_ready & 0xfeU)))) {
        Verilated::overWidthError("out_ready");
    }
}
#endif  // VL_DEBUG
