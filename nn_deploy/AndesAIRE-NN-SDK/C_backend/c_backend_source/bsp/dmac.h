/*
 * Copyright (c) 2012-2019 Andes Technology Corporation
 * All rights reserved.
 *
 */

#ifndef __DMAC_H__
#define __DMAC_H__
//extern int global_benchtime;
//extern int global_cycle_begin;
//extern int global_cycle_end;
/*
 * Exported functions
 */
#include <ae350.h>
#include <stdint.h>

typedef struct atcdmac300_lld_t
{
        volatile uint32_t ctrl;
        volatile uint32_t size;
        volatile uint32_t src_addr_l;
        volatile uint32_t src_addr_h;
        volatile uint32_t dst_addr_l;
        volatile uint32_t dst_addr_h;
        volatile uint32_t LLPL;
        volatile uint32_t LLPH;
} atcdmac300_lld_s;

//extern void dump_dmac300_info(void);
//extern void reset_dmac300(void);
extern void data_copy_dmac300(void *pSrc, void *pDst, int nSize);
extern void data_copy_dmac300_chain_tx(void *w_src, void *w_dst, int w_size, void *b_src, void *b_dst, int b_size);

extern volatile int dma_completed;

//void online_dma_setting(DMA_CHANNEL_REG *dma_ch, void *pSrc, int nSize);
//int onthefly_dma_setting(DMA_CHANNEL_REG *dma_ch, void *pDst);

#define DMA_CHANNEL(n)  ((DMA_CHANNEL_REG *)&(AE350_DMA->CHANNEL[n]))

#define BURST01     0  /* 1 tranfer per burst */
#define BURST02     1  /* 2 tranfer per burst */
#define BURST04     2  /* 4 tranfer per burst */
#define BURST08     3  /* 8 tranfer per burst */
#define BURST16     4  /* 16 tranfer per burst */
#define BURST32     5  /* 32 tranfer per burst */
#define BURST64     6  /* 64 tranfer per burst */
#define BURST128    7  /* 128 tranfer per burst */
#define BURST256    8  /* 256 tranfer per burst */
#define BURST512    9  /* 512 tranfer per burst */
#define BURST1024   10 /* 1024 tranfer per burst */

#define WIDTH_BYTE           0  /* memory syste is byte width */
#define WIDTH_HALFWORD       1  /* memory system is 2 bytes width */
#define WIDTH_WORD           2  /* memory system is 4 bytes width */
#define WIDTH_DOUBLEWORD     3  /* memory system is 8 bytes width */
#define WIDTH_QUADWORD       4  /* memory system is 16 bytes width */
#define WIDTH_EIGHTWORD      5  /* memory system is 32 bytes width */

#define MODE_NORMAL          0  /* memory without H/W handshaking mechanism */
#define MODE_HANDSHAKE       1  /* memory with H/W handshaking mechanism */

#define ADDRESS_INC          0 /* increment address */
#define ADDRESS_DEC          1 /* decrement address */
#define ADDRESS_FIX          2 /* fixed address */
//=================

/* ------------------------------------------------------------- */
#define DMA_CHAN       3  /* DMA Channel number, 0 ~ 7 */
#define SRC_BUS_IND    0  /* source bus interface index, 0 or 1 */
#define DST_BUS_IND    0  /* destination bus interface index, 0 or 1 */
#define CHN_PRIORITY   0 /* channel priority, 0: low, 1: high */
#define SRC_BURST_SIZE BURST1024 /* transfer number per burst, BURST01, BURST02 ... BURST1024 */
#define SRC_WIDTH      WIDTH_DOUBLEWORD  /* source data width, byte, word, double_word ... */
#define DST_WIDTH      WIDTH_DOUBLEWORD  /* source data width, byte, word, double_word ... */
#define SRC_MODE       MODE_NORMAL /* source memory system with/without handshake mechanism */
#define DST_MODE       MODE_NORMAL /* source memory system with/without handshake mechanism */
#define SRC_ADDR_CTRL  ADDRESS_INC /* source address, increment, decrement, fixed */
#define DST_ADDR_CTRL  ADDRESS_INC /* source address, increment, decrement, fixed */
/* ------------------------------------------------------------- */
//#define TIMES
//#define TIMES_GLOBAL
#endif	// __DMAC_H__
