#include <gpmr/GPMRGPUConfig.h>
#include <cudacpp/Runtime.h>
#include <cstdio>

const int MAX_BLOCKS = 32768;
const int NUM_THREADS = 256;

#if 0
__global__ void wordCountReducerKernel(const int blockOffset,
                                       const int numKeys,
                                       const int * const keys,
                                       const int * const vals,
                                       const int * const valOffsets,
                                       const int * const numVals,
                                       int * const outputKeys,
                                       int * const outputVals)
{
  const int index = (blockOffset + blockIdx.x) * blockDim.x + threadIdx.x;
  if (index > numKeys) return;
  const int valCount = numVals[index];
  const int * ptr = vals + valOffsets[index];
  int ret = 0;
  for (int i = 0; i < valCount; ++i) ret += *(ptr++);
  outputKeys[index] = keys[index];
  outputVals[index] = ret;
}

#else
const int THREADS_PER_WARP = 16;
const int WARPS_PER_BLOCK = NUM_THREADS / THREADS_PER_WARP;

__global__ void wordCountReducerKernel(const int numKeys,
                                       const int * const keys,
                                       const int * const vals,
                                       const int * const valOffsets,
                                       const int * const numVals,
                                       int * const outputKeys,
                                       int * const outputVals)
{
  __shared__ volatile int counts[NUM_THREADS];
  __shared__ int nv;
  const int warpIndex = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / THREADS_PER_WARP;
  const int threadWarpIndex = threadIdx.x % THREADS_PER_WARP;
  if (threadIdx.x == 0) nv = *numVals;
  __syncthreads();
  const int cyclesPerWarp = (nv + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
  for (int index = warpIndex; index < numKeys; index += gridDim.x * WARPS_PER_BLOCK)
  {
    counts[threadIdx.x] = 0;
    for (int cycle = 0; cycle < cyclesPerWarp; ++cycle)
    {
      if (threadWarpIndex + cycle * THREADS_PER_WARP < nv) 
      {
        counts[threadIdx.x] += *(vals + valOffsets[index] + threadWarpIndex + cycle * THREADS_PER_WARP);
      }
    }
    if (threadWarpIndex < 8)
    {
      counts[threadIdx.x] += counts[threadIdx.x + 8];
      counts[threadIdx.x] += counts[threadIdx.x + 4];
      counts[threadIdx.x] += counts[threadIdx.x + 2];
      counts[threadIdx.x] += counts[threadIdx.x + 1];
    }
    if (threadWarpIndex == 0)
    {
      outputKeys[index] = keys[index];
      outputVals[index] = vals[index];
    }
  }
}
#endif

void wordCountReducerExecute(const int numKeys,
                             const void * const keys,
                             const void * const vals,
                             const int * const valOffsets,
                             const int * const numVals,
                             GPMRGPUConfig & gpuConfig,
                             cudaStream_t & stream)
{ 
  wordCountReducerKernel<<<60, NUM_THREADS, 0, stream>>>(numKeys,
                                                         reinterpret_cast<const int * >(keys),
                                                         reinterpret_cast<const int * >(vals),
                                                         valOffsets,
                                                         numVals,
                                                         reinterpret_cast<int * >(gpuConfig.keySpace),
                                                         reinterpret_cast<int * >(gpuConfig.valueSpace));
}
