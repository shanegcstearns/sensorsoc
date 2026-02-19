// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtop__pch.h"
#include "Vtop.h"
#include "Vtop___024root.h"

// FUNCTIONS
Vtop__Syms::~Vtop__Syms()
{

    // Tear down scope hierarchy
    __Vhier.remove(0, &__Vscope_takethree);

}

Vtop__Syms::Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
    // Check resources
    Verilated::stackCheck(292);
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    // Setup scopes
    __Vscope_TOP.configure(this, name(), "TOP", "TOP", "<null>", 0, VerilatedScope::SCOPE_OTHER);
    __Vscope_takethree.configure(this, name(), "takethree", "takethree", "takethree", -9, VerilatedScope::SCOPE_MODULE);

    // Set up scope hierarchy
    __Vhier.add(0, &__Vscope_takethree);

    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
        __Vscope_TOP.varInsert(__Vfinal,"clk", &(TOP.clk), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0,0);
        __Vscope_TOP.varInsert(__Vfinal,"cosine_q8", &(TOP.cosine_q8), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"hr_q8", &(TOP.hr_q8), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"hr_rmssd_q8", &(TOP.hr_rmssd_q8), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"in_ready", &(TOP.in_ready), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0,0);
        __Vscope_TOP.varInsert(__Vfinal,"in_valid", &(TOP.in_valid), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0,0);
        __Vscope_TOP.varInsert(__Vfinal,"logit0_q16", &(TOP.logit0_q16), false, VLVT_UINT32,VLVD_OUT|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"logit1_q16", &(TOP.logit1_q16), false, VLVT_UINT32,VLVD_OUT|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"movement_q8", &(TOP.movement_q8), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_ready", &(TOP.out_ready), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_valid", &(TOP.out_valid), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0,0);
        __Vscope_TOP.varInsert(__Vfinal,"rstn", &(TOP.rstn), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0,0);
        __Vscope_TOP.varInsert(__Vfinal,"state_out", &(TOP.state_out), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"B1", &(TOP.takethree__DOT__B1), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1,1 ,0,15 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"B2", &(TOP.takethree__DOT__B2), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1,1 ,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"H1", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__H1))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"SCALE_Q0", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__SCALE_Q0))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"SCALE_Q1", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__SCALE_Q1))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"SCALE_Q2", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__SCALE_Q2))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"SCALE_Q3", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__SCALE_Q3))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_DONE", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_DONE))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_IDLE", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_IDLE))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L1_INIT", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L1_INIT))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L1_MAC0", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L1_MAC0))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L1_MAC1", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L1_MAC1))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L1_MAC2", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L1_MAC2))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L1_MAC3", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L1_MAC3))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L1_WRITE", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L1_WRITE))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L2_0_DONE", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L2_0_DONE))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L2_0_INIT", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L2_0_INIT))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L2_0_MAC", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L2_0_MAC))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L2_1_DONE", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L2_1_DONE))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L2_1_INIT", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L2_1_INIT))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"S_L2_1_MAC", const_cast<void*>(static_cast<const void*>(&(TOP.takethree__DOT__S_L2_1_MAC))), true, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"W1", &(TOP.takethree__DOT__W1), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2,1 ,0,15 ,0,3 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"W2", &(TOP.takethree__DOT__W2), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2,1 ,0,1 ,0,15 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"acc", &(TOP.takethree__DOT__acc), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"clk", &(TOP.takethree__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"cosine_q8", &(TOP.takethree__DOT__cosine_q8), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"hidden", &(TOP.takethree__DOT__hidden), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1,1 ,0,15 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"hr_q8", &(TOP.takethree__DOT__hr_q8), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"hr_rmssd_q8", &(TOP.takethree__DOT__hr_rmssd_q8), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"i", &(TOP.takethree__DOT__i), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"ii", &(TOP.takethree__DOT__ii), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"in_ready", &(TOP.takethree__DOT__in_ready), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"in_valid", &(TOP.takethree__DOT__in_valid), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"j", &(TOP.takethree__DOT__j), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"jj", &(TOP.takethree__DOT__jj), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"l0_r", &(TOP.takethree__DOT__l0_r), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"l1_r", &(TOP.takethree__DOT__l1_r), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"logit0", &(TOP.takethree__DOT__logit0), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"logit0_q16", &(TOP.takethree__DOT__logit0_q16), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"logit1_q16", &(TOP.takethree__DOT__logit1_q16), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,0,1 ,31,0);
        __Vscope_takethree.varInsert(__Vfinal,"movement_q8", &(TOP.takethree__DOT__movement_q8), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"out_ready", &(TOP.takethree__DOT__out_ready), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"out_valid", &(TOP.takethree__DOT__out_valid), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"rstn", &(TOP.takethree__DOT__rstn), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"st", &(TOP.takethree__DOT__st), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,1 ,3,0);
        __Vscope_takethree.varInsert(__Vfinal,"state_out", &(TOP.takethree__DOT__state_out), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"state_r", &(TOP.takethree__DOT__state_r), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0,0);
        __Vscope_takethree.varInsert(__Vfinal,"x0", &(TOP.takethree__DOT__x0), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"x1", &(TOP.takethree__DOT__x1), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"x2", &(TOP.takethree__DOT__x2), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
        __Vscope_takethree.varInsert(__Vfinal,"x3", &(TOP.takethree__DOT__x3), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,0,1 ,15,0);
    }
}
