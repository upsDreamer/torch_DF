/*
 * Copyright (c) 2012-2019 Andes Technology Corporation
 * All rights reserved.
 *
 */

#include "ae350.h"
#include "dmac.h"
#include "platform.h"
#include <stdint.h>
#include "cache.h"
#include "andla_hw.h"

#define UART_OUT 0
#if UART_OUT

#include <stdio.h>
static char str[0x100];
#define printf(arg...) do { sprintf(str, ##arg); uart_puts(str); } while (0)

#endif

DMA_CHANNEL_REG* dma_ch3;
volatile int dma_completed;

//static unsigned int dma_ctl_reg;
#define DMA_CTL_REG ( \
        (SRC_BUS_IND << 31) | \
        (DST_BUS_IND << 30) | \
        (CHN_PRIORITY << 29) | \
        (SRC_BURST_SIZE << 24) |\
        (SRC_WIDTH << 21) | \
        (DST_WIDTH << 18) | \
        (SRC_MODE << 17)  | \
        (DST_MODE << 16)  | \
        (SRC_ADDR_CTRL << 14)  | \
        (DST_ADDR_CTRL << 12)  | \
        (0x0 << 8 ) | \
        (0x0 << 4 ) | \
        (0x0 << 3 ) | \
        (0x0 << 2 ) | \
        (0x0 << 1 ) | \
        (0x1 << 0 ) )  

#define DMA_CTL_REG_0 ( \
        (SRC_BUS_IND << 31) | \
        (DST_BUS_IND << 30) | \
        (CHN_PRIORITY << 29) | \
        (SRC_BURST_SIZE << 24) |\
        (SRC_WIDTH << 21) | \
        (DST_WIDTH << 18) | \
        (SRC_MODE << 17)  | \
        (DST_MODE << 16)  | \
        (SRC_ADDR_CTRL << 14)  | \
        (DST_ADDR_CTRL << 12)  | \
        (0x0 << 8 ) | \
        (0x0 << 4 ) | \
        (0x1 << 3 ) | \
        (0x1 << 2 ) | \
        (0x1 << 1 ) | \
        (0x1 << 0 ) )  

#define DMA_CHAIN_TX 1
#if DMA_CHAIN_TX
// linked list descriptor
#if 0 // move to dmac.h
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
#endif
atcdmac300_lld_s llp_desp[1]  __attribute__ ((aligned(32)));
volatile int dma_chain_tx_completed = 0;
#endif

#if 0
int pow2(int ind)
{
	int ret = 1;

	while (ind > 0 )
	{
		ret = ret * 2;
		ind --;
	}
	return ret;
}

void dump_dmac300_info(void)
{
	unsigned int reg;

	reg = DEV_DMA->IDREV;

	//uart_puts("[Dump DMA controller control registers]\n");
	printf("[ae350p] DMA IP version: 0x%06x, major version: 0x%01x, minor version: 0x%01x\n", \
          (reg & 0xFFFFFF00) >> 8, /* ID number for DMAC */\
          (reg & 0x000000F0) >> 4, /* Major revision number */\
          (reg & 0x0000000F) >> 0);  /* Minor revision number */

	reg = DEV_DMA->DMACFG;
	printf("[ae350p] DMAC configuration: \n\t Chain transfer support: %d, Request Synchronization: %d,\n"
        "\tAXI data width %d bits, AXI address width %d bits,\n"
        "\tDMA Core number: %d, AXI bus: %d,\n"
        "\tReq/Ack: %d, FIFO depth %d, DMA channel number: %d\n",
			((reg & 0x80000000) >> 31)? 1 : 0, /* Chain transfer */
			((reg & 0x40000000) >> 30)? 1 : 0, /* Request synchronization */
			(32 * pow2(((reg & 0x03000000) >> 24)+1)), /* AXI data width */
			((reg & 0x00FE0000)>>17) +24,       /* AXI address width */
			((reg & 0x00010000)>>16) +1,         /* DMA core number */
			((reg & 0x00008000)>>15) +1,         /* AXI Buse number */
			((reg & 0x00007C00)>>10),            /* Req/Ack pair number */
			((reg & 0x000003F0)>>4),             /* FIFO depth */
			(reg & 0x0000000F));                 /* Channel number */
}

void print_dma_status()
{
   printf("dma_ch3->TRANSIZE=%d\n", dma_ch3->TRANSIZE);
   printf("llp_desp[0].size=%d\n", llp_desp[0].size);
   printf("llp_desp[1].size=%d\n", llp_desp[1].size);
   printf("DEV_DMA->INTSTATUS=%x\n", DEV_DMA->INTSTATUS);
}
#endif

void reset_dmac300(void)
{
    DEV_DMA->DMACTRL = 1; /* reset the DMA core and disable all channels */
    DEV_DMA->INTSTATUS = 0xFFFFFFFF; /* clean all interrupt status register */
}

void data_copy_dmac300(void *pSrc, void *pDst, int nSize)
{
    //DMA_CHANNEL_REG* dma_ch3;

    DEV_DMA->DMACTRL = 1; /* reset the DMA core and disable all channels */
    DEV_DMA->INTSTATUS = 0xFFFFFFFF; /* clean all interrupt status register */

    dma_ch3 = DMA_CHANNEL(DMA_CHAN);
    dma_ch3->TRANSIZE = (nSize + ((1<<WIDTH_DOUBLEWORD)-1)) >> WIDTH_DOUBLEWORD;
    dma_ch3->SRCADDRL = (uint32_t)(uintptr_t) pSrc; /* low part of source address */
    dma_ch3->SRCADDRH = 0x0;                 /* high part of source address */
    dma_ch3->DSTADDRH = 0x0;                 /* high part of destination address */
    dma_ch3->LLPL     = 0x0;                 /* low part of Linked List Pointer */
    dma_ch3->LLPH     = 0x0;                 /* high part of Linked List Pointer */
    dma_ch3->DSTADDRL = (uint32_t)(uintptr_t) pDst; /* low part of destination address */

    dma_completed = 0;
    dma_ch3->CTRL = DMA_CTL_REG;

}

#if DMA_CHAIN_TX
void data_copy_dmac300_chain_tx(void *w_src, void *w_dst, int w_size, void *b_src, void *b_dst, int b_size)
{
    DEV_DMA->DMACTRL = 1; /* reset the DMA core and disable all channels */
    DEV_DMA->INTSTATUS = 0xFFFFFFFF; /* clean all interrupt status register */
    dma_ch3 = DMA_CHANNEL(DMA_CHAN);

#if 0
    //llp_desp[0].ctrl = DMA_CTL_REG_0;
    llp_desp[0].size = (w_size + ((1<<WIDTH_DOUBLEWORD)-1)) >> WIDTH_DOUBLEWORD;
    llp_desp[0].src_addr_l = (uint32_t)(uintptr_t) w_src; /* low part of source address */
    llp_desp[0].src_addr_h = 0x0;                 /* high part of source address */
    llp_desp[0].dst_addr_l = (uint32_t)(uintptr_t) w_dst; /* low part of destination address */
    llp_desp[0].dst_addr_h = 0x0;                 /* high part of destination address */
    llp_desp[0].LLPL     = (uint32_t)((uintptr_t)&llp_desp[1] & 0xffffffff);
    llp_desp[0].LLPH     = (uint32_t)((uintptr_t)&llp_desp[1] >> 32);

    llp_desp[1].ctrl = DMA_CTL_REG;
    llp_desp[1].size = (b_size + ((1<<WIDTH_DOUBLEWORD)-1)) >> WIDTH_DOUBLEWORD;
    llp_desp[1].src_addr_l = (uint32_t)(uintptr_t) b_src; /* low part of source address */
    llp_desp[1].src_addr_h = 0x0;                 /* high part of source address */
    llp_desp[1].dst_addr_l = (uint32_t)(uintptr_t) b_dst; /* low part of destination address */
    llp_desp[1].dst_addr_h = 0x0;                 /* high part of destination address */
    llp_desp[1].LLPL       = 0x0;           /* low part of Linked List Pointer */
    llp_desp[1].LLPH       = 0x0;                 /* high part of Linked List Pointer */

    dma_ch3->TRANSIZE = llp_desp[0].size;
    dma_ch3->SRCADDRL = llp_desp[0].src_addr_l;
    dma_ch3->SRCADDRH = llp_desp[0].src_addr_h;
    dma_ch3->DSTADDRL = llp_desp[0].dst_addr_l;
    dma_ch3->DSTADDRH = llp_desp[0].dst_addr_h;
    dma_ch3->LLPL     = llp_desp[0].LLPL;
    dma_ch3->LLPH     = llp_desp[0].LLPH;
#else
    llp_desp[0].ctrl = DMA_CTL_REG;
    llp_desp[0].size = (b_size + ((1<<WIDTH_DOUBLEWORD)-1)) >> WIDTH_DOUBLEWORD;
    llp_desp[0].src_addr_l = (uint32_t)(uintptr_t) b_src; /* low part of source address */
    llp_desp[0].src_addr_h = 0x0;                 /* high part of source address */
    llp_desp[0].dst_addr_l = (uint32_t)(uintptr_t) b_dst; /* low part of destination address */
    llp_desp[0].dst_addr_h = 0x0;                 /* high part of destination address */
    llp_desp[0].LLPL       = 0x0;           /* low part of Linked List Pointer */
    llp_desp[0].LLPH       = 0x0;                 /* high part of Linked List Pointer */

    dma_ch3->TRANSIZE = (w_size + ((1<<WIDTH_DOUBLEWORD)-1)) >> WIDTH_DOUBLEWORD;
    dma_ch3->SRCADDRL = (uint32_t)(uintptr_t) w_src;
    dma_ch3->SRCADDRH = 0x00;
    dma_ch3->DSTADDRL = (uint32_t)(uintptr_t) w_dst;
    dma_ch3->DSTADDRH = 0x00;
    dma_ch3->LLPL     = (uint32_t)((uintptr_t)&llp_desp[0] & 0xffffffff);
    dma_ch3->LLPH     = (uint32_t)((uintptr_t)&llp_desp[0] >> 32);
#if BUFFER_ATTR_CACHEABLE
    unsigned long xx = (unsigned long)llp_desp;
    //nds_dcache_flush_range((unsigned long)llp_desp,(unsigned long)(llp_desp+sizeof(llp_desp)));
    nds_dcache_flush_range(xx,xx+sizeof(llp_desp));
#endif
#endif

    dma_completed = 0;
    dma_ch3->CTRL = DMA_CTL_REG_0;
    //dma_ch3->CTRL = DMA_CTL_REG_0 | 0x01;

}
#endif

void dma_irq_handler(void)
{
    unsigned int reg;

    reg = DEV_DMA->INTSTATUS;
//printf("YEH dma_irq_handler: reg=%x\n", reg);
    DEV_DMA->INTSTATUS = reg;

    dma_completed = 1;
}

