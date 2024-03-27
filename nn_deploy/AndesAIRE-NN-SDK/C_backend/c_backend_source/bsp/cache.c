#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "platform.h"

#ifdef __nds32__
#include <nds32_intrinsic.h>
#elif defined __riscv
#include <nds_intrinsic.h>
#endif

int icache_detection(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_ICM_CFG);

	return ((tmp32 >> 6) & 0x7);
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MICM_CFG);

	return ((tmp32 >> 6) & 0x7);
#endif
	return 0;
}

int dcache_detection(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_DCM_CFG);

	return ((tmp32 >> 6) & 0x7);
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MDCM_CFG);

	return ((tmp32 >> 6) & 0x7);
#endif
	return 0;
}

int get_icache_status(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_CACHE_CTL);

	return (tmp32 & 0x1);
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MCACHE_CTL);

	return (tmp32 & 0x1);
#endif
	return 0;
}

int get_dcache_status(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_CACHE_CTL);

	return ((tmp32 >> 1) & 0x1);
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MCACHE_CTL);

	return ((tmp32 >> 1) & 0x1);
#endif
	return 0;
}

void enable_icache(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_MMU_CTL);
	tmp32 = (tmp32 & (~0x6)) | 0x4;
	__nds32__mtsr(tmp32, NDS32_SR_MMU_CTL);

	tmp32 = __nds32__mfsr(NDS32_SR_CACHE_CTL);
	tmp32 |= 0x1;
	__nds32__mtsr(tmp32, NDS32_SR_CACHE_CTL);
	__nds32__isb();
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MCACHE_CTL);
	tmp32 |= 0x1;
	__nds__mtsr(tmp32, NDS_MCACHE_CTL);
	__asm__ __volatile__ ("fence.i");
#endif
}

void enable_dcache(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_MMU_CTL);
	tmp32 = (tmp32 & (~0x6)) | 0x4;
	__nds32__mtsr(tmp32, NDS32_SR_MMU_CTL);

	tmp32 = __nds32__mfsr(NDS32_SR_CACHE_CTL);
	tmp32 |= 0x2;
	__nds32__mtsr(tmp32, NDS32_SR_CACHE_CTL);
	__nds32__isb();
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MCACHE_CTL);
	tmp32 |= 0x2;
	__nds__mtsr(tmp32, NDS_MCACHE_CTL);
	__asm__ __volatile__ ("fence.i");
#endif
}

void disable_icache(void) {

#ifdef __nds32__
	uint32_t tmp32;
	tmp32 = __nds32__mfsr(NDS32_SR_CACHE_CTL);
	tmp32 &= ~0x1;
	__nds32__mtsr(tmp32, NDS32_SR_CACHE_CTL);
	__nds32__isb();
#elif defined __riscv
	unsigned long tmp32;
	tmp32 = __nds__mfsr(NDS_MCACHE_CTL);
	tmp32 &= ~(unsigned long)0x1;
	__nds__mtsr(tmp32, NDS_MCACHE_CTL);
	__asm__ __volatile__ ("fence.i");
#endif
}

void disable_dcache(void) {

#ifdef __nds32__
	uint32_t tmp32;
	uint32_t l2c_exist;

	// L2 cache detection
	l2c_exist = (__nds32__mfsr(NDS32_SR_MSC_CFG) >> 9) & 0x1;

#ifdef NDS32_BASELINE_V3
	if (l2c_exist == 0)
		__nds32__cctl_l1d_wball_one_lvl();
	else
		__nds32__cctl_l1d_wball_alvl();

	__nds32__msync_all();
	__nds32__cctl_l1d_invalall();
#else
	if (l2c_exist == 0)
		__asm__ __volatile__ (".word 0xe1010064");	// CCTL L1D_WBALL, one level
	else
		__asm__ __volatile__ (".word 0xe1050064");	// CCTL L1D_WBALL, all level

	__asm__ __volatile__ ("msync all");
	__asm__ __volatile__ (".word 0xe1000064");		// CCTL L1D_INVALALL
#endif

	tmp32 = __nds32__mfsr(NDS32_SR_CACHE_CTL);
	tmp32 &= ~0x2;
	__nds32__mtsr(tmp32, NDS32_SR_CACHE_CTL);
	__nds32__isb();
#elif defined __riscv
	unsigned long tmp32;
	__asm__ __volatile__ ("fence.i");

	tmp32 = __nds__mfsr(NDS_MCACHE_CTL);
	tmp32 &= ~(unsigned long)0x2;
	__nds__mtsr(tmp32, NDS_MCACHE_CTL);
#endif
}

void set_ntc(uint32_t ntc_num, uint32_t value) {
#ifdef __nds32__
	uint32_t tmp32;

	tmp32 = __nds32__mfsr(NDS32_SR_MMU_CTL);
	tmp32 &= ~0x6;
	tmp32 |= (value << (2 * ntc_num + 1));

	__nds32__mtsr(tmp32, NDS32_SR_MMU_CTL);
#else
	UNUSED(ntc_num);
	UNUSED(value);
#endif
}

void set_ntm(uint32_t ntm_num, uint32_t value) {
#ifdef __nds32__
	uint32_t tmp32;

	tmp32 = __nds32__mfsr(NDS32_SR_MMU_CTL);
	tmp32 &= ~0x6;
	tmp32 |= (value << (2 * ntm_num + 11));

	__nds32__mtsr(tmp32, NDS32_SR_MMU_CTL);
#else
	UNUSED(ntm_num);
	UNUSED(value);
#endif
}
/*
 * Copyright (c) 2012-2018 Andes Technology Corporation
 * All rights reserved.
 *
 */

#include "cache.h"

/* AndeStar_V5_SPA_UMxxx_V0.1.18-20180525.pdf, page 167 */
#define NDS_MMSC_CFG				0xFC2	//Machine CSRs
#define NDS_MCACHE_CTL				0x7CA	//Machine CSRs
#define NDS_MCCTLBEGINADDR			0x7CB	//Machine CSRs
#define NDS_MCCTLCOMMAND			0x7CC	//Machine CSRs
#define NDS_MCCTLDATA				0x7CD	//Machine CSRs
#define NDS_UCCTLBEGINADDR			0x80B	//User CSRs
#define NDS_UCCTLCOMMAND			0x80C	//User CSRs
#define NDS_SCCTLDATA				0x9CD	//Supervisor CSRs

/* AndeStar CCTL Register Bit Field */
#define CCTL_SUEN_MSK				(1ULL << (8))		//NDS_MCACHE_CTL
#define CCTLCSR_MSK				(1ULL << (16))		//NDS_MMSC_CFG

/* AndeStar CCTL Command Machine mode */
#define CCTL_L1D_VA_INVAL			0	//Allow S/U mode
#define CCTL_L1D_VA_WB				1	//Allow S/U mode
#define CCTL_L1D_VA_WBINVAL			2	//Allow S/U mode
#define CCTL_L1D_VA_LOCK			3
#define CCTL_L1D_VA_UNLOCK			4
#define CCTL_L1D_WBINVAL_ALL		6
#define CCTL_L1D_WB_ALL				7
#define CCTL_L1I_VA_INVAL			8	//Allow S/U mode
#define CCTL_L1I_VA_LOCK			11
#define CCTL_L1I_VA_UNLOCK			12
#define CCTL_L1D_IX_INVAL			16
#define CCTL_L1D_IX_WB				17
#define CCTL_L1D_IX_WBINVAL			18
#define CCTL_L1D_IX_RTAG			19
#define CCTL_L1D_IX_RDATA			20
#define CCTL_L1D_IX_WTAG			21
#define CCTL_L1D_IX_WDATA			22
#define CCTL_L1D_INVAL_ALL			23
#define CCTL_L1I_IX_INVAL			24
#define CCTL_L1I_IX_RTAG			27
#define CCTL_L1I_IX_RDATA			28
#define CCTL_L1I_IX_WTAG			29
#define CCTL_L1I_IX_WDATA			30

#define MEMSET(s, c, n) __builtin_memset ((s), (c), (n))

static inline void GIE_SAVE(unsigned long *var) {
	*var = read_csr(NDS_MSTATUS);
	/* Disable global interrupt for core */
	clear_csr(NDS_MSTATUS, MSTATUS_MIE);
}

static inline void GIE_RESTORE(unsigned long var) {
	if (var & MSTATUS_MIE) {
		/* Enable global interrupt for core */
		set_csr(NDS_MSTATUS, MSTATUS_MIE);
	}
}

void nds_dcache_invalidate(void) {
	unsigned long saved_gie=0;
	GIE_SAVE(&saved_gie);

	write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_INVAL_ALL);

	GIE_RESTORE(saved_gie);
}

void nds_dcache_writeback_range(unsigned long start, unsigned long size) {
	unsigned long saved_gie=0;
	unsigned long line_size;
	unsigned long end = start + size;
	line_size = CACHE_LINE_SIZE(DCACHE);
	end = (end + line_size - 1) & (~(line_size-1));

	GIE_SAVE(&saved_gie);

	while (end > start) {
		write_csr(NDS_MCCTLBEGINADDR, start);
		write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_VA_WB);
		start += line_size;
	}

	GIE_RESTORE(saved_gie);
}

/*
 * nds_dcache_invalidate_range(start, end)
 *
 * throw away all D-cached data in specified region without an obligation
 * to write them back.
 *
 * Note: however that we must clean the D-cached entries around the
 * boundaries if the start and/or end address are not cache aligned.
 */
void nds_dcache_invalidate_range(unsigned long start, unsigned long size) {
	unsigned long saved_gie=0;
	unsigned long line_size;
	unsigned long end = start + size;
	line_size = CACHE_LINE_SIZE(DCACHE);
	end = (end + line_size - 1) & (~(line_size-1));

	GIE_SAVE(&saved_gie);

	while (end > start) {
		write_csr(NDS_MCCTLBEGINADDR, start);
		write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_VA_INVAL);
		start += line_size;
	}

	GIE_RESTORE(saved_gie);
}

void nds_dcache_flush_range(unsigned long start, unsigned long size) {
	unsigned long saved_gie=0;
	unsigned long line_size;
	unsigned long end = start + size;
	line_size = CACHE_LINE_SIZE(DCACHE);
	end = (end + line_size - 1) & (~(line_size-1));

	GIE_SAVE(&saved_gie);

	while (end > start) {
		write_csr(NDS_MCCTLBEGINADDR, start);
		write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_VA_WBINVAL);
		start += line_size;
	}

	GIE_RESTORE(saved_gie);
}

static inline __attribute__((always_inline)) void nds_dcache_invalidate_line(unsigned long addr) {
	unsigned long saved_gie=0;

	GIE_SAVE(&saved_gie);

	write_csr(NDS_MCCTLBEGINADDR, addr);
	write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_VA_INVAL);

	GIE_RESTORE(saved_gie);
}

static inline __attribute__((always_inline)) void nds_dcache_flush_line(unsigned long addr) {
	unsigned long saved_gie;

	GIE_SAVE(&saved_gie);

	write_csr(NDS_MCCTLBEGINADDR, addr);
	write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_VA_WBINVAL);

	GIE_RESTORE(saved_gie);
}

static inline __attribute__((always_inline)) void unaligned_cache_line_move(unsigned char* src, unsigned char* dst, unsigned long len) {
	unsigned long i;
	unsigned char* src_p = src;
	unsigned char* dst_p = dst;

	for (i = 0; i < len; ++i)
		*(dst_p+i)=*(src_p+i);
}

static void nds_dcache_invalidate_partial_line(unsigned long start, unsigned long end) {
	unsigned long line_size;
	line_size = CACHE_LINE_SIZE(DCACHE);
	unsigned char buf[line_size];

	unsigned long aligned_start = start & (~(line_size-1));
	unsigned long aligned_end   = (end + line_size - 1) & (~(line_size-1));
	unsigned long end_offset    = end & (line_size-1);

	/* handle cache line unaligned */
	if (aligned_start < start) {
		unaligned_cache_line_move((unsigned char*)aligned_start, buf, start - aligned_start);
	}
	if (end < aligned_end) {
		unaligned_cache_line_move((unsigned char*)end, buf + end_offset, aligned_end - end);
	}

	nds_dcache_invalidate_line(start);

	/* handle cache line unaligned */
	if (aligned_start < start) {
		unaligned_cache_line_move(buf, (unsigned char*)aligned_start, start - aligned_start);
	}
	if (end < aligned_end) {
		unaligned_cache_line_move(buf + end_offset, (unsigned char*)end, aligned_end - end);
	}
}

void nds_dma_writeback_range(unsigned long start, unsigned long end) {
	unsigned long line_size;
	line_size = CACHE_LINE_SIZE(DCACHE);
	start = start & (~(line_size-1));
	end = (end + line_size - 1) & (~(line_size-1));
	if (start == end)
		return;

	nds_dcache_writeback_range(start, end);
}

void nds_dma_invalidate_range(unsigned long start, unsigned long end) {
	unsigned long line_size = CACHE_LINE_SIZE(DCACHE);

	unsigned long aligned_start = (start + line_size - 1) & (~(line_size - 1));
	unsigned long aligned_end   = end & (~(line_size - 1));

	if (aligned_start > aligned_end) {
		nds_dcache_flush_line(start);
	}
	else {
		if (start < aligned_start) {
			nds_dcache_flush_line(start);
		}
		if (aligned_start < aligned_end) {
			nds_dcache_invalidate_range(aligned_start, aligned_end-1);
		}
		if (aligned_end < end) {
			nds_dcache_flush_line(end);
		}
	}
}

void nds_dma_invalidate_boundary(unsigned long start, unsigned long end) {
	unsigned long line_size = CACHE_LINE_SIZE(DCACHE);

	unsigned long aligned_start = (start + line_size - 1) & (~(line_size - 1));
	unsigned long aligned_end   = end & (~(line_size - 1));

	if (aligned_start > aligned_end) {
		nds_dcache_invalidate_partial_line(start, end);
	}
	else {
		if (start < aligned_start) {
			nds_dcache_invalidate_partial_line(start, aligned_start);
		}
		if (aligned_end < end) {
			nds_dcache_invalidate_partial_line(aligned_end, end);
		}
	}
}

/*
 * nds_icache_invalidate(void)
 *
 * invalidate all I-cached data.
 *
 * Note: CCTL doesn't support icache invalidate all operation,
 * so this function emulates "Invalidate all" operation by invalidating 
 * each cache line of cache (index-based CCTL).
 */
void nds_icache_invalidate(void) {
	unsigned long saved_gie=0;
	unsigned long line_size;
	unsigned long end;
	unsigned long i;

	line_size = CACHE_LINE_SIZE(ICACHE);
	end = CACHE_WAY(ICACHE) * CACHE_SET(ICACHE) * line_size;

	GIE_SAVE(&saved_gie);

	for (i = 0; i < end; i += line_size) {
		write_csr(NDS_MCCTLBEGINADDR, i);
		write_csr(NDS_MCCTLCOMMAND, CCTL_L1I_IX_INVAL);
	}

	GIE_RESTORE(saved_gie);
}

#define ISET_MSK                                0x7
#define IWAY_MSK                                0x38
#define ISIZE_MSK                               0x1C0
#define DSET_MSK                                0x7
#define DWAY_MSK                                0x38
#define DSIZE_MSK                               0x1C0

/* AndeStar CCTL Register Bit Field */
/* CSR NDS_MCACHE_CTL */
#define CCTL_SUEN_MSK                           (1ULL << (8))
/* CSR NDS_MMSC_CFG */
#define CCTLCSR_MSK                             (1ULL << (16))
#define VCCTL_MSK                               ((1ULL << (18)) | (1ULL << (19)))

/* SMU.SYSTEMCFG Configuration Register */
#define L2C_CTL_OFF                             8
#define L2C_CTL_MSK                             (0x1 << L2C_CTL_OFF)

/* Configuration Register */
#define L2C_SIZE_OFF                            7
#define L2C_SIZE_MSK                            (0x1F << L2C_SIZE_OFF)
#define L2C_SIZE_0KB                            (0x00 << L2C_SIZE_OFF)
#define L2C_SIZE_128KB                          (0x01 << L2C_SIZE_OFF)
#define L2C_SIZE_256KB                          (0x02 << L2C_SIZE_OFF)
#define L2C_SIZE_512KB                          (0x04 << L2C_SIZE_OFF)
#define L2C_SIZE_1024KB                         (0x08 << L2C_SIZE_OFF)
#define L2C_SIZE_2048KB                         (0x10 << L2C_SIZE_OFF)
#define L2C_LINE_SIZE                           32

/* Control Register */
#define L2C_ENABLE                              0x1

unsigned int iset, iway, isize;
unsigned int dset, dway, dsize;

int enable_cache()
{
        /* Check whether the CPU configured with L1C cache. */
        if (!((read_csr(NDS_MICM_CFG) & ISIZE_MSK) >> 6) || !((read_csr(NDS_MDCM_CFG) & DSIZE_MSK) >> 6)) {
                return 0;
        }

        /* Enable L1C cache */
        write_csr(NDS_MCACHE_CTL, (read_csr(NDS_MCACHE_CTL) | 0x3));

#if 0 // AnDLA doesn't support L2 currently
        /* Check whether the CPU configured with L2C cache. */
        if (DEV_SMU->SYSTEMCFG[0] & L2C_CTL_MSK) {
                /* Enable L2C cache */
                DEV_L2CACHE->CTL |= L2C_ENABLE;
        }
#endif
        return 1;
}

int dump_idcache_info(void)
{
        unsigned long icm_cfg = 0, dcm_cfg = 0;
        //unsigned int l2c_size = 0;

#if 0
        if( !enable_cache())
        {
                printf("CPU does not support I/D Cache\n");
                return -1;
        }
#endif
        /* Check if support CCTL feature */
        if (read_csr(NDS_MMSC_CFG) & CCTLCSR_MSK)
        {
                printf("CPU supports CCTL operation\n");
                if (read_csr(NDS_MMSC_CFG) & VCCTL_MSK)
                {
                        printf("CPU supports CCTL auto-increment\n");
                }
                else
                {
                        printf("CPU does NOT support CCTL auto-increment\n");
                }
        }
        else
        {
                printf("CPU supports FENCE operation\n");
        }

        /* Get ICache ways, sets, line size */
        icm_cfg = read_csr(NDS_MICM_CFG);
        if ((icm_cfg & ISET_MSK) < 7) {
                iset = (unsigned int)(1 << ((icm_cfg & ISET_MSK) + 6));
        } else {
                iset = 0;
        }
        printf("The L1C ICache sets = %d\n", iset);

        iway = (unsigned int)(((icm_cfg & IWAY_MSK) >> 3) + 1);
        printf("The L1C ICache ways = %d\n", iway);

        if (((icm_cfg & ISIZE_MSK) >> 6) && (((icm_cfg & ISIZE_MSK) >> 6) <= 5)) {
                isize = (unsigned int)(1 << (((icm_cfg & ISIZE_MSK) >> 6) + 2));
        } else if (((icm_cfg & ISIZE_MSK) >> 6) >= 6) {
                printf("Warning L1C i cacheline size is reserved value\n");
                isize = 0;
        } else {
                isize = 0;
        }

        printf("The L1C ICache line size = %d\n", isize);
        if (isize == 0) {
                printf("This CPU doesn't have L1C ICache.\n");
                return -1;
        } 
#if 0
        else {
                /* Enable L1C ICache */
                write_csr(NDS_MCACHE_CTL, (read_csr(NDS_MCACHE_CTL) | 0x1));
        }
#endif
        /* Get DCache ways, sets, line size  */
        dcm_cfg = read_csr(NDS_MDCM_CFG);
        if ((dcm_cfg & DSET_MSK) < 7) {
                dset = (unsigned int)(1 << ((dcm_cfg & DSET_MSK) + 6));
        } else {
                dset = 0;
        }
        printf("The L1C DCache sets = %d\n", dset);

        dway = (unsigned int)(((dcm_cfg & DWAY_MSK) >> 3) + 1);
        printf("The L1C DCache ways = %d\n", dway);

        if (((dcm_cfg & DSIZE_MSK) >> 6) && (((dcm_cfg & DSIZE_MSK) >> 6) <= 5)) {
                dsize = (unsigned int)(1 << (((dcm_cfg & DSIZE_MSK) >> 6) + 2));
        } else if (((dcm_cfg & DSIZE_MSK) >> 6) >= 6) {
                printf("Warning L1C d cacheline size is reserved value\n");
                dsize = 0;
        } else {
                dsize = 0;
        }

        printf("The L1C DCache line size = %d\n", dsize);
        if (dsize == 0) {
                printf("This CPU doesn't have L1C DCache.\n");
                return -1;
        } 
#if 0
        else {
                /* Enable L1C DCache */
                write_csr(NDS_MCACHE_CTL, (read_csr(NDS_MCACHE_CTL) | 0x2));
        }
#endif
        if (read_csr(NDS_MCACHE_CTL) & 0x1) {
                printf("Enable L1C I cache\n");
        }

        if (read_csr(NDS_MCACHE_CTL) & 0x2) {
                printf("Enable L1C D cache\n");
        }

        if (!(read_csr(NDS_MCACHE_CTL) & 0x3)) {
                printf("Can't enable L1C I/D cache\n");
                return -1;
        }

#if 0 // AnDLA doesn't support L2 currently
        if (DEV_SMU->SYSTEMCFG[0] & L2C_CTL_MSK) {
                l2c_size = 0;
                printf("CPU supports L2C cache\n");

                if ((DEV_L2CACHE->CFG & L2C_SIZE_MSK) >> L2C_SIZE_OFF) {
                        l2c_size = (unsigned int)(1 << (((DEV_L2CACHE->CFG & L2C_SIZE_MSK) >> L2C_SIZE_OFF) + 6));
                }

                printf("The L2C cache size = %dKB\n", l2c_size);

                /* Enable L2C Cache */
                DEV_L2CACHE->CTL |= L2C_ENABLE;
        } else {
                printf("CPU does NOT support L2C cache\n");
        }
#endif

        return 0;
}

void invalidate_cache(void)
{
        __nds__fence(FENCE_RW, FENCE_RW);
}

void dcache_flush(void)
{
        if (read_csr(NDS_MMSC_CFG) & CCTLCSR_MSK) {
                /* L1C DCache writeback and invalidate all */
                write_csr(NDS_MCCTLCOMMAND, CCTL_L1D_WBINVAL_ALL);

#if 0 // AnDLA doesn't support L2 currently
                if (*(volatile unsigned int *)SMU_SYSTEMCFG_REG & L2C_CTL_MSK) {
                        /* L2C DCache writeback and invalidate all */
                        *(volatile unsigned long long *)CCTL_L2_CMD_REG(0) = CCTL_L2_WBINVAL_ALL;
                }
#endif
        } else {
                /* FENCE.I go data writeback w/o data invalid but instruction invalid.
                   As src code is copied to dst and go to execute dst instruction,
                   you should use FENCE.I instead of FENCE */
                __nds__fencei();
        }
}

