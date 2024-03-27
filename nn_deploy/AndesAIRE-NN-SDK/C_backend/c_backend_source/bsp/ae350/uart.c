/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */

#include "platform.h"

#define BAUD_RATE(n)            (UCLKFREQ / (n) / 16)

int uart_init(unsigned int baudrate)
{
	/* Set DLAB to 1 */
	DEV_UART2->LCR |= 0x80;

	/* Set DLL for baudrate */
	DEV_UART2->DLL = (BAUD_RATE(baudrate) >> 0) & 0xff;
	DEV_UART2->DLM = (BAUD_RATE(baudrate) >> 8) & 0xff;

	/* LCR: Length 8, no parity, 1 stop bit. */
	DEV_UART2->LCR = 0x03;

	/* FCR: Enable FIFO, reset TX and RX. */
	DEV_UART2->FCR = 0x07;
	
	return 0;
}

int uart_getc(void)
{
#define SERIAL_LSR_RDR          0x01
	while ((DEV_UART2->LSR & SERIAL_LSR_RDR) == 0) ;

	return (int)DEV_UART2->RBR;
}

void uart_putc(int c)
{
#define SERIAL_LSR_THRE         0x20
	while ((DEV_UART2->LSR & SERIAL_LSR_THRE) == 0) ;

	DEV_UART2->THR = (unsigned int)c;
}

int uart_puts(const char *s)
{
	int len = 0;

	while (*s) {
		uart_putc(*s);

		if (*s == '\n')
			uart_putc('\r');
		s++;
		len++;
	}
	return len;
}
/* move to common/iochar.c
int outbyte(int c)
{
	uart_putc(c);
	if (c =='\n')
		uart_putc('\r');
	return c;
}
*/
#if 0
/*
 * Retarget functions for printf()
 */
#include <sys/stat.h>

int _fstat (int file, struct stat * st) {
	UNUSED(file);
	UNUSED(st);
	return 0;
}

int _write (int file, char * ptr, int len) {
	UNUSED(file);
	//extern void uart_putc(int c);
	int i;

	for (i = 0; i < len; i++)
	{
		if (*ptr == '\n')
			uart_putc('\r');
		uart_putc((int)*ptr++);
	}
	return len;
}

int _read (int file, char * ptr, int len) {
	UNUSED(file);
	//extern void uart_putc(int c);
	//extern int uart_getc(void);
	char c;
	int i;

	for (i = 0; i < len; i++) {
		c = (char)uart_getc();
		*ptr++ = c;
		if (c == '\r') break;
		//uart_putc(c);
	}
	return (len - i);
}

#endif