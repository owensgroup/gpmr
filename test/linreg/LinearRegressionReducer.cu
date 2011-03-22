#include <cstdio>

__global__ void linearRegressionReducerKernel(const int   * const keys,
                                              const float * const vals,
                                              const int * numVals,
                                              int * const keySpace,
                                              float * const valSpace)
{
  __shared__ volatile int nv;
  __shared__ volatile float numElems;
  float fsum = 0.0f;
  if (threadIdx.x == 0) nv = numVals[0];
  for (int i = 0; i < nv; ++i)
  {
    fsum += vals[threadIdx.x * nv + i];
  }
  if (threadIdx.x == blockDim.x - 1) numElems = fsum;
  keySpace[threadIdx.x] = keys[threadIdx.x];
  valSpace[threadIdx.x] = fsum / numElems;
}

void linearRegressionReducerExecute(const int   * const keys,
                                    const float * const vals,
                                    const int * numVals,
                                    int * const keySpace,
                                    float * const valSpace,
                                    cudaStream_t & stream)
{
  linearRegressionReducerKernel<<<1, 6, 0, stream>>>(keys, vals, numVals, keySpace, valSpace);

#if 0
  {
    int cpuKeys[6];
    float cpuVals[6];

    cudaMemcpy(cpuKeys, keySpace, sizeof(cpuKeys), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuVals, valSpace, sizeof(cpuVals), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; ++i)
    {
      printf("%2d: %2d - %f\n", i, cpuKeys[i], cpuVals[i]);
    }
    fflush(stdout);
  }
#endif
}
