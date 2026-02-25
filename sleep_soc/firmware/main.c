#include <stdint.h>

// TEST MMIO
#define TEST_STATUS  (*(volatile uint32_t*)0x0300F000u)
#define TEST_CODE    (*(volatile uint32_t*)0x0300F004u)

// TIMER MMIO (base 0x03002000)
#define TIMER_CTRL    (*(volatile uint32_t*)0x03002000u)  // bit0=enable, bit1=periodic
#define TIMER_RELOAD  (*(volatile uint32_t*)0x03002004u)
#define TIMER_COUNT   (*(volatile uint32_t*)0x03002008u)
#define TIMER_EVENT   (*(volatile uint32_t*)0x0300200Cu)  // W1C bit0

// POWER MMIO (base 0x03001000)
#define PWR_CTRL      (*(volatile uint32_t*)0x03001000u)  // bit0=sleep_req
#define PWR_WAKE_STATUS (*(volatile uint32_t*)0x03001004u)
#define PWR_WAKE_REASON (*(volatile uint32_t*)0x03001008u)

int main(void)
{
    // Set timer reload to 10000 cycles (short for simulation)
    TIMER_RELOAD = 10000;
    TIMER_COUNT  = 10000;

    // Enable timer (one-shot: bit0=enable, bit1=0)
    TIMER_CTRL = 0x1;

    // Request sleep (bit0=1)
    PWR_CTRL = 0x1;

    // CPU clock gets gated here by RTL until timer fires

    // After wake: clear the timer event
    TIMER_EVENT = 0x1;

    // Clear wake status
    PWR_WAKE_STATUS = 0xFFFFFFFF;

    // Signal PASS
    TEST_STATUS = 0xCAFEBABE;

    while (1);
}