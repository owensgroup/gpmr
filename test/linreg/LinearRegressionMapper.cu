#include <cstdio>

const int NUM_THREADS = 256;
const int NUM_CATEGORIES = 5;

template <typename T, int NUM_ARRS>
__device__ void multiReduce(volatile T arr[NUM_ARRS][NUM_THREADS])
{
  if (threadIdx.x < 128)
  {
    for (int i = 0; i < NUM_ARRS; ++i) arr[i][threadIdx.x] += arr[i][threadIdx.x + 128];
  }
  __syncthreads();
  if (threadIdx.x < 64)
  {
    for (int i = 0; i < NUM_ARRS; ++i) arr[i][threadIdx.x] += arr[i][threadIdx.x + 64];
  }
  __syncthreads();
  if (threadIdx.x < 32)
  {
    for (int i = 0; i < NUM_ARRS; ++i)
    {
      arr[i][threadIdx.x] = arr[i][threadIdx.x] + arr[i][threadIdx.x + 32];
      arr[i][threadIdx.x] = arr[i][threadIdx.x] + arr[i][threadIdx.x + 16];
      arr[i][threadIdx.x] = arr[i][threadIdx.x] + arr[i][threadIdx.x +  8];
      arr[i][threadIdx.x] = arr[i][threadIdx.x] + arr[i][threadIdx.x +  4];
      arr[i][threadIdx.x] = arr[i][threadIdx.x] + arr[i][threadIdx.x +  2];
      arr[i][threadIdx.x] = arr[i][threadIdx.x] + arr[i][threadIdx.x +  1];
    }
  }
  __syncthreads();
}

__global__ void linearRegressionMapperZero(float * const valueSpace)
{
  valueSpace[threadIdx.x] = 0.0f;
}

__global__ void linearRegressionMapperKernel(const float2 * const points,
                                             const int numPoints,
                                             float * const valueSpace)
{
  __shared__ volatile float vals[NUM_CATEGORIES][NUM_THREADS];
  float x = 0.0f, y = 0.0f, xx = 0.0f, yy = 0.0f, xy = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numPoints; i += gridDim.x * blockDim.x)
  {
    float2 myPoint = points[i];
    x  += myPoint.x;
    y  += myPoint.y;
    xx += myPoint.x * myPoint.x;
    yy += myPoint.y * myPoint.y;
    xy += myPoint.x * myPoint.y;
  }
  vals[0][threadIdx.x] = x;
  vals[1][threadIdx.x] = y;
  vals[2][threadIdx.x] = xx;
  vals[3][threadIdx.x] = yy;
  vals[4][threadIdx.x] = xy;
  __syncthreads();
  multiReduce<float, NUM_CATEGORIES>(vals);
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < NUM_CATEGORIES; ++i) valueSpace[blockIdx.x * NUM_CATEGORIES + i] = vals[i][0];
  }
}

__global__ void linearRegressionMapperCombineKernel(const int numElems,
                                                    const int numBlocks,
                                                    int * const keySpace,
                                                    float * const valSpace,
                                                    const float * const globalValueSpace)
{
  float val = globalValueSpace[threadIdx.x];
  for (int i = 1; i < numBlocks; ++i)
  {
    val += globalValueSpace[threadIdx.x + i * blockDim.x];
  }
  keySpace[threadIdx.x] = threadIdx.x;
  valSpace[threadIdx.x] += val;
  keySpace[NUM_CATEGORIES] = NUM_CATEGORIES;
  valSpace[NUM_CATEGORIES] += static_cast<float>(numElems);
}

void linearRegressionMapperExecute(const void * const points,
                                   const int numPoints,
                                   const int numBlocks,
                                   const int numThreads,
                                   void * const keySpace,
                                   void * const valueSpace,
                                   void * const globalValueSpace,
                                   const bool firstMapping,
                                   cudaStream_t & stream)
{
  if (firstMapping) linearRegressionMapperZero<<<1, numBlocks, 0, stream>>>(reinterpret_cast<float * >(valueSpace));
  linearRegressionMapperKernel<<<numBlocks, numThreads, 0, stream>>>(reinterpret_cast<const float2 * >(points),
                                                                     numPoints,
                                                                     reinterpret_cast<float * >(globalValueSpace));
  linearRegressionMapperCombineKernel<<<1, NUM_CATEGORIES, 0, stream>>>(numPoints,
                                                                        numBlocks,
                                                                        reinterpret_cast<int * >(keySpace),
                                                                        reinterpret_cast<float * >(valueSpace),
                                                                        reinterpret_cast<float * >(globalValueSpace));
}
