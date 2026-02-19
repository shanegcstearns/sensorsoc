// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"

// Parameter definitions for Vtop___024root
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_IDLE;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L1_INIT;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L1_MAC0;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L1_MAC1;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L1_MAC2;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L1_MAC3;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L1_WRITE;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L2_0_INIT;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L2_0_MAC;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L2_0_DONE;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L2_1_INIT;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L2_1_MAC;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_L2_1_DONE;
constexpr CData/*3:0*/ Vtop___024root::takethree__DOT__S_DONE;
constexpr IData/*31:0*/ Vtop___024root::takethree__DOT__H1;
constexpr IData/*31:0*/ Vtop___024root::takethree__DOT__SCALE_Q0;
constexpr IData/*31:0*/ Vtop___024root::takethree__DOT__SCALE_Q1;
constexpr IData/*31:0*/ Vtop___024root::takethree__DOT__SCALE_Q2;
constexpr IData/*31:0*/ Vtop___024root::takethree__DOT__SCALE_Q3;


void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf);

Vtop___024root::Vtop___024root(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop___024root___ctor_var_reset(this);
}

void Vtop___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtop___024root::~Vtop___024root() {
}
