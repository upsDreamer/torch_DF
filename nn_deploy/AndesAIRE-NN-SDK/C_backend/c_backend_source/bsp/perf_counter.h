#ifndef __PERF_COUNTER_H__
#define __PERF_COUNTER_H__

#include "platform.h"
//#include <nds_intrinsic.h>
//#include "core_v5.h"
/*
 * The mcycle counter is 64-bit counter. But RV32 access
 * it as two 32-bit registers, so we check for rollover
 * with this routine as suggested by the RISC-V Priviledged
 * Architecture Specification.
 */
__attribute__((always_inline))
static inline unsigned long long rdmcycle(void)
{
#if __riscv_xlen == 32
        do {
                unsigned long hi = read_csr(NDS_MCYCLEH);
                unsigned long lo = read_csr(NDS_MCYCLE);

                if (hi == read_csr(NDS_MCYCLEH))
                        return ((unsigned long long)hi << 32) | lo;
        } while(1);
#else
        return read_csr(NDS_MCYCLE);
#endif
}

/*
 * The minstret counter is 64-bit counter. But RV32 access
 * it as two 32-bit registers, same as for mcycle.
 */
__attribute__((always_inline))
static inline unsigned long long rdminstret(void)
{
#if __riscv_xlen == 32
        do {
                unsigned long hi = read_csr(NDS_MINSTRETH);
                unsigned long lo = read_csr(NDS_MINSTRET);

                if (hi == read_csr(NDS_MINSTRETH))
                        return ((unsigned long long)hi << 32) | lo;
        } while(1);
#else
        return read_csr(NDS_MINSTRET);
#endif
}

extern void reset_perf_counter(void);
extern void get_perf_counter(unsigned long long *icache_access, unsigned long long *icache_miss, 
	unsigned long long *dcache_access, unsigned long long *dcache_miss);

#endif // __PERF_COUNTER_H__
