cmake_minimum_required(VERSION 3.7)

add_library(bsp STATIC
	start.S
	ae350/ae350.c
	ae350/interrupt.c
	ae350/uart.c
	cache.c
	initfini.c
	reset.c
	trap.c
	)

target_include_directories(bsp PRIVATE ${CMAKE_SOURCE_DIR})
