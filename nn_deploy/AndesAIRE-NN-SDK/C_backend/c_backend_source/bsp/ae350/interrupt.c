/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */

#include <stdio.h>
#include "platform.h"

#define NESTED_VPLIC_COMPLETE_INTERRUPT(irq)            \
do {                                                    \
	clear_csr(NDS_MIE, MIP_MEIP);                   \
	__nds__plic_complete_interrupt(irq);            \
	__asm volatile("fence io, io");                 \
	set_csr(NDS_MIE, MIP_MEIP);                     \
} while(0)

void default_irq_handler(void)
{
	printf("Default interrupt handler\n");
}


void wdt_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void rtc_period_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void rtc_alarm_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void pit_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void spi1_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void spi2_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void i2c_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void gpio_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void uart1_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void uart2_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void dma_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void swint_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void ac97_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void sdc_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void mac_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void lcd_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void touch_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void standby_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
void wakeup_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));
//void andla_irq_handler(void) __attribute__((weak, alias("default_irq_handler")));

#if(USE_ANDLA == 1)
extern void andla_irq_handler(void);
#endif

void __attribute__ ((interrupt)) entry_irq1(void)
{
	NESTED_IRQ_ENTER();

	rtc_period_irq_handler();               // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_RTCPERIOD_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq2(void)
{
	NESTED_IRQ_ENTER();

	rtc_alarm_irq_handler();                // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_RTCALARM_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq3(void)
{
	NESTED_IRQ_ENTER();

	pit_irq_handler();                      // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_PIT_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq4(void)
{
	NESTED_IRQ_ENTER();

	spi1_irq_handler();                     // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_SPI1_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq5(void)
{
	NESTED_IRQ_ENTER();

	spi2_irq_handler();                     // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_SPI2_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq6(void)
{
	NESTED_IRQ_ENTER();

	i2c_irq_handler();                      // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_I2C_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq7(void)
{
	NESTED_IRQ_ENTER();

	gpio_irq_handler();                     // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_GPIO_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq8(void)
{
	NESTED_IRQ_ENTER();

	uart1_irq_handler();                    // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_UART1_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq9(void)
{
	NESTED_IRQ_ENTER();

	uart2_irq_handler();                    // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_UART2_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq10(void)
{
	NESTED_IRQ_ENTER();

	dma_irq_handler();                      // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_DMA_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq12(void)
{
	NESTED_IRQ_ENTER();

	swint_irq_handler();                    // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_SWINT_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq17(void)
{
	NESTED_IRQ_ENTER();

	ac97_irq_handler();                     // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_AC97_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq18(void)
{
	NESTED_IRQ_ENTER();

	sdc_irq_handler();                      // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_SDC_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq19(void)
{
	NESTED_IRQ_ENTER();

	mac_irq_handler();                      // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_MAC_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq20(void)
{
	NESTED_IRQ_ENTER();

	lcd_irq_handler();                      // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_LCDC_SOURCE);

	NESTED_IRQ_EXIT();
}

/*AnDLA interrupt*/
void __attribute__ ((interrupt)) entry_irq21(void)
{
	NESTED_IRQ_ENTER();

#if(USE_ANDLA == 1)
	andla_irq_handler();                      // Call ISR
#endif    

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_ANDLA_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq25(void)
{
	NESTED_IRQ_ENTER();

	touch_irq_handler();                    // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_TOUCH_SOURCE);

	NESTED_IRQ_EXIT();
}

void __attribute__ ((interrupt)) entry_irq26(void)
{
	NESTED_IRQ_ENTER();

	standby_irq_handler();                  // Call ISR

	NESTED_VPLIC_COMPLETE_INTERRUPT(IRQ_STANDBY_SOURCE);

	NESTED_IRQ_EXIT();
}

/* It is supposed to be the highest priority interrupt */
void __attribute__ ((interrupt)) entry_irq27(void)
{
	wakeup_irq_handler();                   // Call ISR

	__nds__plic_complete_interrupt(IRQ_WAKEUP_SOURCE);
}
