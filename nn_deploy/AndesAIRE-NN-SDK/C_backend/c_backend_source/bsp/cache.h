
#ifndef __CACHE_H
#define __CACHE_H

#include <inttypes.h>
#include "general.h"

#include "core_v5.h"

#define NDS_ISET                        0x7
#define NDS_IWAY                        0x38
#define NDS_ISIZE                       0x1C0
#define NDS_DSET                        0x7
#define NDS_DWAY                        0x38
#define NDS_DSIZE                       0x1C0

enum cache_t{ICACHE, DCACHE};

static inline unsigned long CACHE_SET(enum cache_t cache) {
	if(cache == ICACHE)
		return ((read_csr(NDS_MICM_CFG) & NDS_ISET) < 7) ? 
			(unsigned long)(1 << ((read_csr(NDS_MICM_CFG) & NDS_ISET) + 6)) : 0;
	else
		return ((read_csr(NDS_MDCM_CFG) & NDS_DSET) < 7) ? 
			(unsigned long)(1 << ((read_csr(NDS_MDCM_CFG) & NDS_DSET) + 6)) : 0;
}

static inline unsigned long CACHE_WAY(enum cache_t cache) {
	if(cache == ICACHE)
		return (unsigned long)(((read_csr(NDS_MICM_CFG) & NDS_IWAY) >> 3) + 1);
	else
		return (unsigned long)(((read_csr(NDS_MDCM_CFG) & NDS_DWAY) >> 3) + 1);
}

static inline unsigned long CACHE_LINE_SIZE(enum cache_t cache) {
	if(cache == ICACHE)
		return (((read_csr(NDS_MICM_CFG) & NDS_ISIZE) >> 6) && (((read_csr(NDS_MICM_CFG) & NDS_ISIZE) >> 6) <= 5)) ?
			(unsigned long)(1 << (((read_csr(NDS_MICM_CFG) & NDS_ISIZE) >> 6) + 2)) : 0;
	else
		return (((read_csr(NDS_MDCM_CFG) & NDS_DSIZE) >> 6) && (((read_csr(NDS_MDCM_CFG) & NDS_DSIZE) >> 6) <= 5)) ?
			(unsigned long)(1 << (((read_csr(NDS_MDCM_CFG) & NDS_DSIZE) >> 6) + 2)) : 0;
}

/* dcache ops */
extern void nds_dcache_invalidate(void);

extern void nds_dcache_writeback_range(unsigned long start, unsigned long size);
extern void nds_dcache_invalidate_range(unsigned long start, unsigned long size);
extern void nds_dcache_flush_range(unsigned long start, unsigned long size);

/* DMA-specific ops */
extern void nds_dma_writeback_range(unsigned long start, unsigned long end);
extern void nds_dma_invalidate_range(unsigned long start, unsigned long end);
extern void nds_dma_invalidate_boundary(unsigned long start, unsigned long end);

/* icache ops */
extern void nds_icache_invalidate(void);

extern int icache_detection(void);
extern int dcache_detection(void);
extern int get_icache_status(void);
extern int get_dcache_status(void);
extern void enable_icache(void);
extern void enable_dcache(void);
extern void disable_icache(void);
extern void disable_dcache(void);
extern void set_ntc(uint32_t ntc_num, uint32_t value);
extern void set_ntm(uint32_t ntm_num, uint32_t value);

extern int dump_idcache_info(void);
extern void invalidate_cache(void);
void dcache_flush(void);
#define dcache_flush_range(start, end)      nds_dcache_flush_range(start, end)
#endif // __CACHE_H
