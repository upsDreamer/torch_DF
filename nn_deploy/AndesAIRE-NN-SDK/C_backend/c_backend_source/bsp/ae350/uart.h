/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */
#ifndef __UART_H__
#define __UART_H__

/*
 * Exported functions
 */
extern int uart_init(unsigned int baudrate);
extern int uart_getc(void);
extern void uart_putc(int c);
extern int uart_puts(const char *s);
extern int outbyte(int c);

#endif	// __UART_H__
