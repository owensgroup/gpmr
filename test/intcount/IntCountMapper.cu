#include <gpmr/GPMRGPUFunctions.h>

__global__ void intCountMapper(const int numElems, const int * const elems, GPMRGPUConfig gpmrGPUConfig)
{
  int * keys = reinterpret_cast<int * >(gpmrGPUConfig.keySpace);
  int * vals = reinterpret_cast<int * >(gpmrGPUConfig.valueSpace);
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numElems; index += gridDim.x * blockDim.x)
  {
    keys[index] = elems[index];
    vals[index] = 1;
  }
}

void intCountMapperExecute(const int numElems,
                            GPMRGPUConfig & gpmrGPUConfig,
                            void * const gpuMemoryForChunk,
                            cudaStream_t & kernelStream)
{
  intCountMapper<<<60, 512, 0, kernelStream>>>(numElems, reinterpret_cast<int * >(gpuMemoryForChunk), gpmrGPUConfig);
  if (0)
  {
    int * cpuKeys = new int[40];
    int * cpuVals = new int[40];
    cudaMemcpy(cpuKeys, gpmrGPUConfig.keySpace,   sizeof(int) * 40, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuVals, gpmrGPUConfig.valueSpace, sizeof(int) * 40, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 40; ++i)
    {
      printf("%10d => %2d\n", cpuKeys[i], cpuVals[i]);
    }
    fflush(stdout);
  }
}
