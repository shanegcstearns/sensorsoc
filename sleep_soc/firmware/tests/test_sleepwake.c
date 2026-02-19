#include <stdint.h>

// ----------------------
// Base addresses
// ----------------------
#define TIMER_BASE     0x03002000u
#define PWR_BASE       0x03001000u
#define TEST_BASE      0x0300F000u

// ----------------------
// timer_mmio offsets
// ----------------------
#define T_CTRL         (TIMER_BASE + 0x0u)  // bit0 enable, bit1 periodic
#define T_RELOAD       (TIMER_BASE + 0x4u)
#define T_COUNT        (TIMER_BASE + 0x8u)
#define T_EVENT        (TIMER_BASE + 0xCu)  // W1C bit0, read shows latched

// ----------------------
// pwrctrl_mmio offsets
// ----------------------
#define P_CTRL         (PWR_BASE + 0x0u)    // bit0 sleep_req
#define P_WAKE_STATUS  (PWR_BASE + 0x4u)    // R/W1C latched wake flags
#define P_WAKE_REASON  (PWR_BASE + 0x8u)    // RO snapshot on wake

// ----------------------
// test_mmio offsets
// ----------------------
#define TEST_STATUS    (TEST_BASE + 0x0u)
#define TEST_CODE      (TEST_BASE + 0x4u)

#define TEST_PASS      0xCAFEBABEu
#define TEST_FAIL      0xDEADBEEFu

// Wake bits per your soc_top: bit0=timer, bit1=ml
#define WAKE_TIMER_BIT (1u << 0)

// ----------------------
// MMIO helpers
// ----------------------
static inline void mmio_write32(uint32_t addr, uint32_t data) {
    *(volatile uint32_t *)addr = data;
}

static inline uint32_t mmio_read32(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

// ----------------------
// PASS/FAIL helpers
// ----------------------
static void pass(void) {
    mmio_write32(TEST_STATUS, TEST_PASS);
    while (1) { } // halt
}

static void fail(uint32_t code) {
    mmio_write32(TEST_CODE, code);
    mmio_write32(TEST_STATUS, TEST_FAIL);
    while (1) { } // halt
}

int main(void) {
    // 1) Clear any old wake flags / timer event
    mmio_write32(P_WAKE_STATUS, 0xFFFFFFFFu); // W1C: clear all latched bits
    mmio_write32(T_EVENT, 1u);                // clear timer event latch

    // 2) Program a short one-shot timer
    mmio_write32(T_RELOAD, 50u);
    mmio_write32(T_COUNT,  20u);
    mmio_write32(T_CTRL,   0x1u);             // enable=1, periodic=0

    // 3) Ask SoC to sleep (cpu clock should gate off)
    mmio_write32(P_CTRL, 1u);

    // 4) Wait for wake (timer event). If CPU sleeps, execution pauses here and
    // resumes after wake ungates the clock. If CPU never sleeps, we just spin.
    while ((mmio_read32(T_EVENT) & 1u) == 0u) {
        __asm__ volatile ("nop");
    }

    // 5) After wake, check what happened
    uint32_t reason = mmio_read32(P_WAKE_REASON);
    uint32_t status = mmio_read32(P_WAKE_STATUS);
    uint32_t tev    = mmio_read32(T_EVENT) & 1u;

    // Timer should have fired
    if (!tev) fail(0x10);

    // Wake snapshot should include timer (either in reason or status)
    if (((reason | status) & WAKE_TIMER_BIT) == 0u) fail(0x11);

    // 6) Clear event + wake status and clear sleep request so we don't re-sleep
    mmio_write32(T_EVENT, 1u);
    mmio_write32(P_WAKE_STATUS, 0xFFFFFFFFu);
    mmio_write32(P_CTRL, 0u);

    // 7) Report PASS
    pass();
    return 0;
}
