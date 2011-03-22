#ifndef __GPMR_GPMRGPUTHREADFUNCTIONS_H__
#define __GPMR_GPMRGPUTHREADFUNCTIONS_H__

#include <gpmr/GPMRGPUFunctions.h>

template <typename Key, typename Value>
__device__ void gpmrThreadEmitKeyValRegister(const int outputNumber, const Key & key, const Value & value)
{
}

template <typename Key, typename Value>
__device__ void gpmrThreadEmitKeyValShared(const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
}

template <typename Key, typename Value>
__device__ void gpmrThreadEmitKeyValGlobal(const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
}

#endif
