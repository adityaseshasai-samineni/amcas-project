#pragma once
#include <stdint.h>
#define CAP 32768
struct Sample {uint64_t cycles;uint32_t pc,sm,value;};
struct Buffer {uint64_t start;uint32_t count,armed;Sample rows[CAP];};
