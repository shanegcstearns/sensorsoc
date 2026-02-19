// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"

VL_ATTR_COLD void Vtop___024root___eval_static(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_static\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__takethree__DOT__clk__0 
        = vlSelfRef.takethree__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__takethree__DOT__rstn__0 
        = vlSelfRef.takethree__DOT__rstn;
}

VL_ATTR_COLD void Vtop___024root___eval_initial__TOP(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_initial(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_initial\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtop___024root___eval_initial__TOP(vlSelf);
}

VL_ATTR_COLD void Vtop___024root___eval_initial__TOP(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_initial__TOP\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.takethree__DOT__B1[0U] = 0x00000150U;
    vlSelfRef.takethree__DOT__W1[0U][0U] = 0xffc5U;
    vlSelfRef.takethree__DOT__W1[0U][1U] = 0x0020U;
    vlSelfRef.takethree__DOT__W1[0U][2U] = 0x0065U;
    vlSelfRef.takethree__DOT__W1[0U][3U] = 0xffcaU;
    vlSelfRef.takethree__DOT__B1[1U] = 0xffffd4a4U;
    vlSelfRef.takethree__DOT__W1[1U][0U] = 0x0081U;
    vlSelfRef.takethree__DOT__W1[1U][1U] = 0xff94U;
    vlSelfRef.takethree__DOT__W1[1U][2U] = 0x004aU;
    vlSelfRef.takethree__DOT__W1[1U][3U] = 0x0038U;
    vlSelfRef.takethree__DOT__B1[2U] = 0xffff8d88U;
    vlSelfRef.takethree__DOT__W1[2U][0U] = 0xfffbU;
    vlSelfRef.takethree__DOT__W1[2U][1U] = 0xffecU;
    vlSelfRef.takethree__DOT__W1[2U][2U] = 0xffc5U;
    vlSelfRef.takethree__DOT__W1[2U][3U] = 0xfff9U;
    vlSelfRef.takethree__DOT__B1[3U] = 0xffff8363U;
    vlSelfRef.takethree__DOT__W1[3U][0U] = 0xfff8U;
    vlSelfRef.takethree__DOT__W1[3U][1U] = 0xffbaU;
    vlSelfRef.takethree__DOT__W1[3U][2U] = 0x0044U;
    vlSelfRef.takethree__DOT__W1[3U][3U] = 0x0040U;
    vlSelfRef.takethree__DOT__B1[4U] = 0xffffc583U;
    vlSelfRef.takethree__DOT__W1[4U][0U] = 0xffbaU;
    vlSelfRef.takethree__DOT__W1[4U][1U] = 0xffa4U;
    vlSelfRef.takethree__DOT__W1[4U][2U] = 0x004eU;
    vlSelfRef.takethree__DOT__W1[4U][3U] = 0x0030U;
    vlSelfRef.takethree__DOT__B1[5U] = 0x0000d257U;
    vlSelfRef.takethree__DOT__W1[5U][0U] = 0xffd6U;
    vlSelfRef.takethree__DOT__W1[5U][1U] = 0x003cU;
    vlSelfRef.takethree__DOT__W1[5U][2U] = 0x0078U;
    vlSelfRef.takethree__DOT__W1[5U][3U] = 6U;
    vlSelfRef.takethree__DOT__B1[6U] = 0xfffff758U;
    vlSelfRef.takethree__DOT__W1[6U][0U] = 0x0099U;
    vlSelfRef.takethree__DOT__W1[6U][1U] = 0xffe1U;
    vlSelfRef.takethree__DOT__W1[6U][2U] = 0x0018U;
    vlSelfRef.takethree__DOT__W1[6U][3U] = 0xfffaU;
    vlSelfRef.takethree__DOT__B1[7U] = 0xffff67acU;
    vlSelfRef.takethree__DOT__W1[7U][0U] = 0xffbaU;
    vlSelfRef.takethree__DOT__W1[7U][1U] = 0xffb4U;
    vlSelfRef.takethree__DOT__W1[7U][2U] = 0xffcaU;
    vlSelfRef.takethree__DOT__W1[7U][3U] = 0xff9bU;
    vlSelfRef.takethree__DOT__B1[8U] = 0xfffffe23U;
    vlSelfRef.takethree__DOT__W1[8U][0U] = 0xfff5U;
    vlSelfRef.takethree__DOT__W1[8U][1U] = 0x001dU;
    vlSelfRef.takethree__DOT__W1[8U][2U] = 0x0055U;
    vlSelfRef.takethree__DOT__W1[8U][3U] = 0x0010U;
    vlSelfRef.takethree__DOT__B1[9U] = 0xffffbf86U;
    vlSelfRef.takethree__DOT__W1[9U][0U] = 0xff8eU;
    vlSelfRef.takethree__DOT__W1[9U][1U] = 0x0063U;
    vlSelfRef.takethree__DOT__W1[9U][2U] = 0xffe3U;
    vlSelfRef.takethree__DOT__W1[9U][3U] = 0xff82U;
    vlSelfRef.takethree__DOT__B1[0x0aU] = 0x000091b9U;
    vlSelfRef.takethree__DOT__W1[0x0aU][0U] = 0x0059U;
    vlSelfRef.takethree__DOT__W1[0x0aU][1U] = 0xff86U;
    vlSelfRef.takethree__DOT__W1[0x0aU][2U] = 0x0040U;
    vlSelfRef.takethree__DOT__W1[0x0aU][3U] = 0x002dU;
    vlSelfRef.takethree__DOT__B1[0x0bU] = 0x0000a742U;
    vlSelfRef.takethree__DOT__W1[0x0bU][0U] = 0xffb3U;
    vlSelfRef.takethree__DOT__W1[0x0bU][1U] = 0xffa9U;
    vlSelfRef.takethree__DOT__W1[0x0bU][2U] = 0x0028U;
    vlSelfRef.takethree__DOT__W1[0x0bU][3U] = 0xfff4U;
    vlSelfRef.takethree__DOT__B1[0x0cU] = 0xffff908cU;
    vlSelfRef.takethree__DOT__W1[0x0cU][0U] = 0x0012U;
    vlSelfRef.takethree__DOT__W1[0x0cU][1U] = 0xffd4U;
    vlSelfRef.takethree__DOT__W1[0x0cU][2U] = 0xffb8U;
    vlSelfRef.takethree__DOT__W1[0x0cU][3U] = 0xffadU;
    vlSelfRef.takethree__DOT__B1[0x0dU] = 0xffffca5aU;
    vlSelfRef.takethree__DOT__W1[0x0dU][0U] = 0x002aU;
    vlSelfRef.takethree__DOT__W1[0x0dU][1U] = 0xffa4U;
    vlSelfRef.takethree__DOT__W1[0x0dU][2U] = 0xffadU;
    vlSelfRef.takethree__DOT__W1[0x0dU][3U] = 0x0069U;
    vlSelfRef.takethree__DOT__B1[0x0eU] = 0x00007049U;
    vlSelfRef.takethree__DOT__W1[0x0eU][0U] = 0xffe7U;
    vlSelfRef.takethree__DOT__W1[0x0eU][1U] = 0x007eU;
    vlSelfRef.takethree__DOT__W1[0x0eU][2U] = 0xffdbU;
    vlSelfRef.takethree__DOT__W1[0x0eU][3U] = 0xfffcU;
    vlSelfRef.takethree__DOT__B1[0x0fU] = 0x000087ddU;
    vlSelfRef.takethree__DOT__W1[0x0fU][0U] = 0xfff7U;
    vlSelfRef.takethree__DOT__W1[0x0fU][1U] = 0x005bU;
    vlSelfRef.takethree__DOT__W1[0x0fU][2U] = 0x0058U;
    vlSelfRef.takethree__DOT__W1[0x0fU][3U] = 0x0062U;
    vlSelfRef.takethree__DOT__B2[0U] = 0xffff947fU;
    vlSelfRef.takethree__DOT__B2[1U] = 0x00004754U;
    vlSelfRef.takethree__DOT__W2[0U][0U] = 0xffeeU;
    vlSelfRef.takethree__DOT__W2[0U][1U] = 0x0012U;
    vlSelfRef.takethree__DOT__W2[0U][2U] = 0x0011U;
    vlSelfRef.takethree__DOT__W2[0U][3U] = 0x0018U;
    vlSelfRef.takethree__DOT__W2[0U][4U] = 0x0029U;
    vlSelfRef.takethree__DOT__W2[0U][5U] = 0xffe8U;
    vlSelfRef.takethree__DOT__W2[0U][6U] = 0x003eU;
    vlSelfRef.takethree__DOT__W2[0U][7U] = 0x000fU;
    vlSelfRef.takethree__DOT__W2[0U][8U] = 0xfff2U;
    vlSelfRef.takethree__DOT__W2[0U][9U] = 0xffd3U;
    vlSelfRef.takethree__DOT__W2[0U][0x0aU] = 0xfff9U;
    vlSelfRef.takethree__DOT__W2[0U][0x0bU] = 0xffe1U;
    vlSelfRef.takethree__DOT__W2[0U][0x0cU] = 9U;
    vlSelfRef.takethree__DOT__W2[0U][0x0dU] = 0xfff3U;
    vlSelfRef.takethree__DOT__W2[0U][0x0eU] = 0U;
    vlSelfRef.takethree__DOT__W2[0U][0x0fU] = 0x0012U;
    vlSelfRef.takethree__DOT__W2[1U][0U] = 0xffd5U;
    vlSelfRef.takethree__DOT__W2[1U][1U] = 6U;
    vlSelfRef.takethree__DOT__W2[1U][2U] = 0xfff9U;
    vlSelfRef.takethree__DOT__W2[1U][3U] = 0xffe4U;
    vlSelfRef.takethree__DOT__W2[1U][4U] = 0xffd6U;
    vlSelfRef.takethree__DOT__W2[1U][5U] = 0xfff8U;
    vlSelfRef.takethree__DOT__W2[1U][6U] = 0x0019U;
    vlSelfRef.takethree__DOT__W2[1U][7U] = 0xfffeU;
    vlSelfRef.takethree__DOT__W2[1U][8U] = 0x001cU;
    vlSelfRef.takethree__DOT__W2[1U][9U] = 0xffdbU;
    vlSelfRef.takethree__DOT__W2[1U][0x0aU] = 0x003eU;
    vlSelfRef.takethree__DOT__W2[1U][0x0bU] = 0x0035U;
    vlSelfRef.takethree__DOT__W2[1U][0x0cU] = 0xffeaU;
    vlSelfRef.takethree__DOT__W2[1U][0x0dU] = 0xfffdU;
    vlSelfRef.takethree__DOT__W2[1U][0x0eU] = 9U;
    vlSelfRef.takethree__DOT__W2[1U][0x0fU] = 0x0024U;
}

VL_ATTR_COLD void Vtop___024root___eval_final(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_final\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtop___024root___eval_phase__stl(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_settle(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_settle\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00000064U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("/home/jfriday/sensorsoc/ML/takethree/takethree.sv", 15, "", "Settle region did not converge after 100 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
    } while (Vtop___024root___eval_phase__stl(vlSelf));
}

VL_ATTR_COLD void Vtop___024root___eval_triggers__stl(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__stl\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered
                                      [0U]) | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
    vlSelfRef.__VstlFirstIteration = 0U;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
}

VL_ATTR_COLD bool Vtop___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vtop___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vtop___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___trigger_anySet__stl\n"); );
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

void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_stl(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_stl\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD bool Vtop___024root___eval_phase__stl(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__stl\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtop___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = Vtop___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        Vtop___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

bool Vtop___024root___trigger_anySet__ico(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(Vtop___024root___trigger_anySet__ico(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

bool Vtop___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vtop___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge takethree.clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @(negedge takethree.rstn)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ctor_var_reset\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rstn = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5377340664288042355ull);
    vlSelf->in_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2339549897027650563ull);
    vlSelf->in_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1122049356863891575ull);
    vlSelf->movement_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15789365119050159791ull);
    vlSelf->cosine_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6521550443568590044ull);
    vlSelf->hr_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7061345302088192919ull);
    vlSelf->hr_rmssd_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6248206891317926036ull);
    vlSelf->out_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2886291494070200219ull);
    vlSelf->out_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17332470166291283643ull);
    vlSelf->state_out = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16606792248447486632ull);
    vlSelf->logit0_q16 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4107444937973221411ull);
    vlSelf->logit1_q16 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11690772743491364653ull);
    vlSelf->takethree__DOT__clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6783498602888589143ull);
    vlSelf->takethree__DOT__rstn = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10026174030755831305ull);
    vlSelf->takethree__DOT__in_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16335301901325088259ull);
    vlSelf->takethree__DOT__in_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17030161582337456331ull);
    vlSelf->takethree__DOT__movement_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15504098038649568452ull);
    vlSelf->takethree__DOT__cosine_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2642118479006218444ull);
    vlSelf->takethree__DOT__hr_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16161790472935295917ull);
    vlSelf->takethree__DOT__hr_rmssd_q8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4876013281053553227ull);
    vlSelf->takethree__DOT__out_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16947629788650221067ull);
    vlSelf->takethree__DOT__out_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8083434270354295482ull);
    vlSelf->takethree__DOT__state_out = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13580916249695014767ull);
    vlSelf->takethree__DOT__logit0_q16 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5584811504305731816ull);
    vlSelf->takethree__DOT__logit1_q16 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12823037120756257804ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 4; ++__Vi1) {
            vlSelf->takethree__DOT__W1[__Vi0][__Vi1] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10274693316414526560ull);
        }
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->takethree__DOT__B1[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14614845437418396963ull);
    }
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 16; ++__Vi1) {
            vlSelf->takethree__DOT__W2[__Vi0][__Vi1] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7371039098030178301ull);
        }
    }
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->takethree__DOT__B2[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16607413670427381026ull);
    }
    vlSelf->takethree__DOT__ii = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14465253264391598974ull);
    vlSelf->takethree__DOT__jj = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6039784479128260234ull);
    vlSelf->takethree__DOT__x0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2332613515215531394ull);
    vlSelf->takethree__DOT__x1 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7971548714590687077ull);
    vlSelf->takethree__DOT__x2 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2049120500447495326ull);
    vlSelf->takethree__DOT__x3 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2959342944741589962ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->takethree__DOT__hidden[__Vi0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2938569553948240201ull);
    }
    vlSelf->takethree__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8046111466641708098ull);
    vlSelf->takethree__DOT__logit0 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7600925532990008077ull);
    vlSelf->takethree__DOT__l0_r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11422684914239313196ull);
    vlSelf->takethree__DOT__l1_r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2311922821092888100ull);
    vlSelf->takethree__DOT__state_r = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16104350395659972062ull);
    vlSelf->takethree__DOT__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14582727367406796819ull);
    vlSelf->takethree__DOT__j = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7930825197997070041ull);
    vlSelf->takethree__DOT__st = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4505693722819188916ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VstlTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VicoTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggered[__Vi0] = 0;
    }
    vlSelf->__Vtrigprevexpr___TOP__takethree__DOT__clk__0 = 0;
    vlSelf->__Vtrigprevexpr___TOP__takethree__DOT__rstn__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
}
