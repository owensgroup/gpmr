#include <gpmr/GPMRGPUConfig.h>
#include <gpmr/GPMRGPUFunctions.h>

__global__ void intCountReducerKernel(const int numKeys,
                                       const int * const keys,
                                       const int * const vals,
                                       const int * const valOffsets,
                                       const int * const numVals,
                                       GPMRGPUConfig gpuConfig)
{
  __shared__ int outputVals[512];
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numKeys) return;
  int valCount            = numVals[index];
  outputVals[threadIdx.x] = 0;
  int valIndex            = valOffsets[index] - valOffsets[0];
  for (int i = 0; i < valCount; ++i) outputVals[threadIdx.x] += vals[valIndex + i];
  __syncthreads();
  reinterpret_cast<int * >(gpuConfig.keySpace  )[index] = keys[index];
  reinterpret_cast<int * >(gpuConfig.valueSpace)[index] = outputVals[threadIdx.x];
}

__host__ void intCountReducerExecute(const int numKeys,
                                     const int * const keys,
                                     const int * const vals,
                                     const int * const valOffsets,
                                     const int * const numVals,
                                     GPMRGPUConfig & gpuConfig,
                                     cudaStream_t & stream)
{
  dim3 blockSize(512, 1, 1);
  dim3 gridSize((numKeys + blockSize.x - 1) / blockSize.x, 1, 1);
  intCountReducerKernel<<<gridSize, blockSize, 0, stream>>>(numKeys, keys, vals, valOffsets, numVals, gpuConfig);
}
