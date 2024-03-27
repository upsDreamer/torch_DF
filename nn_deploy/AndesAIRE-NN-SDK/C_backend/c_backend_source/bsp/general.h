
#ifndef __GENERAL_H
#define __GENERAL_H

#include <inttypes.h>

// -------------------------------------------------------------------------------
//  Constant definition
// -------------------------------------------------------------------------------
#define FAIL       0
#define SUCCESS    1
#define FALSE      0
#define TRUE       1

#define BIT0       0x1
#define BIT1       0x2
#define BIT2       0x4
#define BIT3       0x8
#define BIT4       0x10
#define BIT5       0x20
#define BIT6       0x40
#define BIT7       0x80
#define BIT8       0x100
#define BIT9       0x200
#define BIT10      0x400
#define BIT11      0x800
#define BIT12      0x1000
#define BIT13      0x2000
#define BIT14      0x4000
#define BIT15      0x8000
#define BIT16      0x10000
#define BIT17      0x20000
#define BIT18      0x40000
#define BIT19      0x80000
#define BIT20      0x100000
#define BIT21      0x200000
#define BIT22      0x400000
#define BIT23      0x800000
#define BIT24      0x1000000
#define BIT25      0x2000000
#define BIT26      0x4000000
#define BIT27      0x8000000
#define BIT28      0x10000000
#define BIT29      0x20000000
#define BIT30      0x40000000
#define BIT31      0x80000000

// -------------------------------------------------------------------------------
//  Data type definition
// -------------------------------------------------------------------------------
typedef void *nds_data_p;
typedef void (*nds_handler_p)(uint32_t intr_no);
typedef void (*nds_intr_setup_p)(uint32_t int_no, uint32_t trigmode, uint32_t triglevel);

typedef struct {
	volatile uint32_t offset[4096]; /* 4K * 4 = SZ_16K */
}__regbase32;

typedef struct {
	volatile uint16_t offset[4096]; /* 4K * 2 = SZ_8K */
}__regbase16;

typedef struct {
	volatile uint8_t offset[4096];  /* 4K * 1 = SZ_4K */
}__regbase8;

// -------------------------------------------------------------------------------
//  Macro define
// -------------------------------------------------------------------------------
// read write any memory space macro
// first find which page you want (&~4095), then find which address you want (&4095)
// ex, if want to access 0x90500048, fist &~4095, get 0x90500000th page, and then &4095 and >>2 (4bytes alignment)
// to find which address you want
#define REG32(a)       ((__regbase32*)(uintptr_t)((a)&~4095))->offset[((a)&4095)>>2]
#define REG16(a)       ((__regbase16*)(uintptr_t)((a)&~4095))->offset[((a)&4095)>>1]
#define REG8(a)        ((__regbase8*)(uintptr_t)((a)&~4095))->offset[((a)&4095)>>0]

#define inb(a)         REG8(a)
#define inhw(a)        REG16(a)
#define inw(a)         REG32(a)

#define outb(a,v)      (REG8(a) = (uint8_t)(uintptr_t)(v))
#define outhw(a,v)     (REG16(a) = (uint16_t)(uintptr_t)(v))
#define outw(a,v)      (REG32(a) = (uint32_t)(uintptr_t)(v))

// Register bit operation macro
#define BIT_MASK(bit_h, bit_l) ((uint32_t)((((uint64_t)0x1 << (1 + bit_h - bit_l)) - (uint32_t)0x1) << bit_l))

#define SET_BIT(var, bit)      do { var |= (0x1 << (bit)); } while(0)
#define CLR_BIT(var, bit)      do { var &= (~(0x1 << (bit))); } while(0)

#define SET_FIELD(var, mask, offset, value)     do {\
	                                              var = ((var) & (~mask)) | (((value) << (offset)) & (mask)); \
                                                } while (0)

#define GET_FIELD(var, mask, offset)            (((var) & (mask)) >> (offset))

#define TEST_FIELD(var, mask)                   ((var) & (mask))

#define CHECK_FIELD(value, mask)                ((value) & (mask))
#define EXTRACT_FIELD(value, mask, offset)      (((value) & (mask)) >> (offset))
#define PREPARE_FIELD(value, mask, offset)      (((value) << (offset)) & (mask))

// Variable bit operation macro
#define VAR_TEST_BIT(var, sig)                  ((var) & (sig))
#define VAR_SET_BIT(var, sig)                   ((var) = (var) | (sig))
#define VAR_CLR_BIT(var, sig)                   ((var) = (var) & (~(sig)))

#ifdef __GNUC__
#define GCC_VERSION                             (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#endif // __GENERAL_H
