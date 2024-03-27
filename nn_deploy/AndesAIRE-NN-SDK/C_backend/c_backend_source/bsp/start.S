/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */
#include "core_v5.h"

	.section .nds_vector, "ax"

	.global reset_vector

reset_vector:
	/* Decide whether this is an NMI or cold reset */
	csrr t0, mcause
	bnez t0, nmi_handler

	.global _start
	.type _start,@function

_start:
	/* Initialize global pointer */
	.option push
	.option norelax
	la gp, __global_pointer$
	.option pop

	/* Initialize stack pointer */
	la t0, _stack
	mv sp, t0

#ifdef __nds_execit
	/* Initialize EXEC.IT table */
	la t0, _ITB_BASE_
	csrw uitb, t0
#endif

#ifdef __riscv_flen
	/* Enable FPU */
	li t0, MSTATUS_FS
	csrrs t0, mstatus, t0

	/* Initialize FCSR */
	fscsr zero
#endif

	/* Initial machine trap-vector Base */
	la t0, __vectors
	csrw mtvec, t0

	/* Enable vectored external PLIC interrupt */
	csrsi mmisc_ctl, 2

	/* Do system low level setup. It must be a leaf function */
	call __platform_init

	/* System reset handler */
	call reset_handler

	/* Infinite loop, if returned accidently */
1:	j 1b

	.weak __platform_init
__platform_init:
	ret

	.weak nmi_handler
nmi_handler:
1:	j 1b

	.text

	.global default_irq_entry
	.align 2

default_irq_entry:
1:	j 1b

	.weak trap_handler
trap_handler:
1:	j 1b

	.macro INTERRUPT num
	.weak entry_irq\num
	.set entry_irq\num, default_irq_entry
	.long entry_irq\num
	.endm

	/* Vector table
	 * NOTE:
	 * The Vector Table base alignment requirement has to be :
	 * " 2^ceiling(log2(N)) x 4 " bytes,
	 * if the PLIC device supports N interrupt sources.
	 */
#define VECTOR_NUMINTRS         32
	.section .vector_table, "a"

	.global __vectors
	.balign 4096

__vectors:
	/* Trap vector */
	.long trap_entry

	/* PLIC interrupt vector */
	.altmacro
	.set irqno, 1
	.rept VECTOR_NUMINTRS
	INTERRUPT %irqno
	.set irqno, irqno+1
	.endr
