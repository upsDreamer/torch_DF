/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */
#include <stdio.h>
#include <stdarg.h>
#include "platform.h"

unsigned long long libnn_total_inst=0;
unsigned long long libnn_total_cycle=0;
/*
 * The mcycle counter is 64-bit counter. But RV32 access
 * it as two 32-bit registers, so we check for rollover
 * with this routine as suggested by the RISC-V Priviledged
 * Architecture Specification.
*/

unsigned int get_timer_freq()
{
	return (unsigned int)CPUFREQ;
}

uint64_t get_timer_value()
{
	#if __riscv_xlen == 32
	do {
		unsigned long hi = read_csr(NDS_MCYCLEH);
		unsigned long lo = read_csr(NDS_MCYCLE);

		if (hi == read_csr(NDS_MCYCLEH))
			return ((uint64_t)hi << 32) | lo;
	} while(1);
	#else
		return (uint64_t)read_csr(NDS_MCYCLE);
	#endif
}

void reset_perf_counter()
{
    // set cache related event counter
    write_csr(NDS_MHPMEVENT3,  0x21); // I_cache access
    write_csr(NDS_MHPMEVENT4,  0x31); // I_cache miss
    write_csr(NDS_MHPMEVENT5,  0x41); // D_cache access
    write_csr(NDS_MHPMEVENT6,  0x51); // D_cache miss

    // clear cache pfm counter
    write_csr(NDS_MHPMCOUNTER3, 0);
#if __riscv_xlen == 32
    write_csr(NDS_MHPMCOUNTER3H, 0);
#endif // CACHE_COUNTER
    write_csr(NDS_MHPMCOUNTER4, 0);
#if __riscv_xlen == 32
    write_csr(NDS_MHPMCOUNTER4H, 0);
#endif // CACHE_COUNTER
    write_csr(NDS_MHPMCOUNTER5, 0);
#if __riscv_xlen == 32
    write_csr(NDS_MHPMCOUNTER5H, 0);
#endif // CACHE_COUNTER
    write_csr(NDS_MHPMCOUNTER6, 0);
#if __riscv_xlen == 32
    write_csr(NDS_MHPMCOUNTER6H, 0);
#endif
    write_csr(NDS_MCYCLE,  0);
    write_csr(NDS_MINSTRET, 0);
#if __riscv_xlen == 32
    write_csr(NDS_MCYCLEH, 0);
    write_csr(NDS_MINSTRETH, 0);
#endif
}

void get_perf_counter(unsigned long long *icache_access, unsigned long long *icache_miss, 
	unsigned long long *dcache_access, unsigned long long *dcache_miss)
{
    // read cache pfm counter
    *icache_access = read_csr(NDS_MHPMCOUNTER3); //read_pfm_event(NDS_MHPMCOUNTER3, NDS_MHPMCOUNTER3H);
    *icache_miss = read_csr(NDS_MHPMCOUNTER4);//read_pfm_event(NDS_MHPMCOUNTER4, NDS_MHPMCOUNTER4H);
    *dcache_access = read_csr(NDS_MHPMCOUNTER5);//read_pfm_event(NDS_MHPMCOUNTER5, NDS_MHPMCOUNTER5H);
    *dcache_miss = read_csr(NDS_MHPMCOUNTER6);//read_pfm_event(NDS_MHPMCOUNTER6, NDS_MHPMCOUNTER6H);
}
