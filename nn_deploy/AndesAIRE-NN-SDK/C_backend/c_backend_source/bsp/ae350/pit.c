/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */

#include "platform.h"

#define PIT_CHNCTRL_CLK_EXTERNAL        (0 << 3)
#define PIT_CHNCTRL_CLK_PCLK            (1 << 3)
#define PIT_CHNCTRL_MODEMASK            0x07
#define PIT_CHNCTRL_TMR_32BIT           1
#define PIT_CHNCTRL_TMR_16BIT           2
#define PIT_CHNCTRL_TMR_8BIT            3
#define PIT_CHNCTRL_PWM                 4
#define PIT_CHNCTRL_MIXED_16BIT         6
#define PIT_CHNCTRL_MIXED_8BIT          7

void pit_init(void)
{
	/* Disable PIT */
	DEV_PIT->CHNEN = 0;

	/* Set PIT control mode */
	DEV_PIT->CHANNEL[0].CTRL = (PIT_CHNCTRL_TMR_32BIT | PIT_CHNCTRL_CLK_PCLK);
	DEV_PIT->CHANNEL[1].CTRL = (PIT_CHNCTRL_TMR_32BIT | PIT_CHNCTRL_CLK_PCLK);
	DEV_PIT->CHANNEL[2].CTRL = (PIT_CHNCTRL_TMR_32BIT | PIT_CHNCTRL_CLK_PCLK);
	DEV_PIT->CHANNEL[3].CTRL = (PIT_CHNCTRL_TMR_32BIT | PIT_CHNCTRL_CLK_PCLK);

	/* Clear and disable interrupt */
	DEV_PIT->INTEN = 0;
	DEV_PIT->INTST = -1;
}

void pit_start(unsigned int ch)
{
	if (ch < 4)
		DEV_PIT->CHNEN |= (0x1 << (4 * (ch)));
}

void pit_stop(unsigned int ch)
{
	if (ch < 4)
		DEV_PIT->CHNEN &= ~(0x1 << (4 * (ch)));
}

unsigned int pit_read(unsigned int ch)
{
	if (ch < 4)
		return	(DEV_PIT->CHANNEL[ch].RELOAD - DEV_PIT->CHANNEL[ch].COUNTER);
	else
		return 0;
}

void pit_set_period(unsigned int ch, unsigned int period)
{
	if (ch < 4)
		DEV_PIT->CHANNEL[ch].RELOAD = period;
}

void pit_irq_enable(unsigned int ch)
{
	if (ch < 4)
		DEV_PIT->INTEN |= (0x1 << (4 * (ch)));
}

void pit_irq_disable(unsigned int ch)
{
	if (ch < 4)
		DEV_PIT->INTEN &= ~(0x1 << (4 * (ch)));
}

void pit_irq_clear(unsigned int ch)
{
	if (ch < 4)
		DEV_PIT->INTST = 0xF << (4 * (ch));
}

unsigned int pit_irq_status(unsigned int ch)
{
	return (DEV_PIT->INTST & (0xF << (4 * (ch))));
}
