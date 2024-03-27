/*
 * Copyright (c) 2012-2021 Andes Technology Corporation
 * All rights reserved.
 *
 */

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

#ifdef __cplusplus
extern "C" {
#endif

#define UNUSED(x) (void)(x)

#include "config.h"

#include "core_v5.h"
#include "ae350.h"
#include "interrupt.h"

#include "uart.h"
#include "gpio.h"
#include "pit.h"

/*****************************************************************************
 * Peripheral device HAL declaration
 ****************************************************************************/
#define DEV_PLMT             AE350_PLMT
#define DEV_DMA              AE350_DMA
#define DEV_SMU              AE350_SMU
#define DEV_UART1            AE350_UART1
#define DEV_UART2            AE350_UART2
#define DEV_PIT              AE350_PIT
#define DEV_RTC              AE350_RTC
#define DEV_GPIO             AE350_GPIO
#define DEV_I2C              AE350_I2C
#define DEV_SPI1             AE350_SPI1
#define DEV_SPI2             AE350_SPI2
#define DEV_L2CACHE          AE350_L2C
//#define DEV_L2C              AE350_L2C

/*****************************************************************************
 * Board specified
 ****************************************************************************/
#define GPIO_USED_MASK       0x7F    /* Which GPIOs to use */

#ifdef __cplusplus
}
#endif

#endif	/* __PLATFORM_H__ */
