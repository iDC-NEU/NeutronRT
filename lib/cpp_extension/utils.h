// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <cinttypes>
#include <string>


/*
GAP Benchmark Suite
Author: Scott Beamer

Miscellaneous helpers that don't fit into classes
*/

/*****************************************************************************
 *                                                                           *
 *   APMA Utility functions                                                  *
 *                                                                           *
 *****************************************************************************/

// Returns the 1-based index of the last (most significant) bit set in x.
inline uint64_t last_bit_set(uint64_t x)
{
  return (sizeof(uint64_t) * 8 - __builtin_clzll(x)); // Linux
}

inline uint64_t floor_log2(uint64_t x)
{
  return (last_bit_set(x) - 1);
}

inline uint64_t ceil_log2(uint64_t x)
{
  assert(x > 0 && "x should not be 0 or less");
  return (last_bit_set(x - 1));
  // i.e. ceil_log2(13) = 4, ceil_log2(27) = 5, etc.
}

inline uint64_t floor_div(uint64_t x, uint64_t y)
{
  return (x / y);
}

inline uint64_t ceil_div(uint64_t x, uint64_t y)
{
  if(x == 0) return 0;
  return (1 + ((x - 1) / y));
}

// Returns the largest power of 2 not greater than x ($2^{\lfloor \lg x \rfloor}$).
inline uint64_t hyperfloor(uint64_t x)
{
  return (1ULL << floor_log2(x));
}

// Returns the smallest power of 2 not less than x ($2^{\lceil \lg x \rceil}$).
inline uint64_t hyperceil(uint64_t x)
{
  return (1ULL << ceil_log2(x));
}

#endif  // UTIL_H_
