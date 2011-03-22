#include <gpmr/GPMRGPUFunctions.h>

const int MAX_CENTERS = 128;
const int NUM_BLOCKS  =  60;
const int NUM_THREADS = 256;
const int NUM_DIMS    =   4;

template <typename T>
static __device__ void reduce(volatile T * const mem)
{
  syncthreads();
  if (threadIdx.x < 128) mem[threadIdx.x] += mem[threadIdx.x + 128]; syncthreads();
  if (threadIdx.x <  64) mem[threadIdx.x] += mem[threadIdx.x +  64]; syncthreads();
  if (threadIdx.x <  32)
  {
    mem[threadIdx.x] += mem[threadIdx.x +  32];
    mem[threadIdx.x] += mem[threadIdx.x +  16];
    mem[threadIdx.x] += mem[threadIdx.x +   8];
    mem[threadIdx.x] += mem[threadIdx.x +   4];
    mem[threadIdx.x] += mem[threadIdx.x +   2];
    mem[threadIdx.x] += mem[threadIdx.x +   1];
  }
  syncthreads();
}

template <typename T>
static __device__ void reduce2(volatile T * const mem)
{
  syncthreads();
  if (threadIdx.x < 28) mem[threadIdx.x] += mem[threadIdx.x + 32];
  syncthreads();
  if (threadIdx.x < 16)
  {
    mem[threadIdx.x] += mem[threadIdx.x +  16];
    mem[threadIdx.x] += mem[threadIdx.x +   8];
    mem[threadIdx.x] += mem[threadIdx.x +   4];
    mem[threadIdx.x] += mem[threadIdx.x +   2];
    mem[threadIdx.x] += mem[threadIdx.x +   1];
  }
  syncthreads();
}

__device__ void loadCenters(float * const sharedCenters, const float * const centers, const int numCenters)
{
  for (int i = threadIdx.x; i < numCenters * NUM_DIMS; i += blockDim.x)
  {
    sharedCenters[i] = centers[i];
  }
  __syncthreads();
}

__device__ void loadVals(volatile float * const dst, const float * const src, const int offset)
{
  for (int i = 0; i < NUM_DIMS; ++i) dst[blockDim.x * i + threadIdx.x] = src[blockDim.x * i + threadIdx.x];
  __syncthreads();
}

__device__ float findDistance(const float * const val, const float * const center)
{
  float dist = 0.0f;
  for (int j = 0; j < NUM_DIMS; ++j)
  {
    const float temp1 = center[j] - val[j];
    dist += temp1 * temp1;
  }
  return dist;
}

__global__ void kmeansMapper(const float * const centers,
                             const int numCenters,
                             float * const dataPoints,
                             const int numElems,
                             float * const accumCenters, // a 2d array, stored as [#centers][#blocks]
                             int   * const accumTotals)  // a 3d array, stored as [#centers][#dims][#blocks]
{
  __shared__          float sharedCenters[MAX_CENTERS * NUM_DIMS];
  __shared__ volatile float vals[NUM_THREADS * NUM_DIMS];
  __shared__ volatile int   keys[NUM_THREADS];
  float val[NUM_DIMS];
  float bestDist;
  int key;

  loadCenters(sharedCenters, centers, numCenters);

  for (int offset = blockIdx.x * blockDim.x; offset < numElems; offset += gridDim.x * blockDim.x)
  {
    loadVals(vals, dataPoints + offset * NUM_DIMS, offset);
    for (int dim = 0; dim < NUM_DIMS; ++dim) val[dim] = vals[threadIdx.x * NUM_DIMS + dim];
    bestDist = 2.0f * NUM_DIMS; // all data dims are in [0,1], so this will always be bigger.

    __syncthreads();

    key = -1;
    for (int center = 0; center < numCenters; ++center)
    {
      float dist = findDistance(val, sharedCenters + center * NUM_DIMS);
      if (dist < bestDist)
      {
        key = center;
        bestDist = dist;
      }
    }
    __syncthreads();
    for (int center = 0; center < numCenters; ++center)
    {
      keys[threadIdx.x] = (key == center ? 1 : 0);
      reduce<int>(keys);
      if (threadIdx.x == 0) accumTotals[center * NUM_BLOCKS + blockIdx.x] += keys[0];
      for (int dim = 0; dim < NUM_DIMS; ++dim)
      {
        vals[threadIdx.x] = (key == center) ? val[dim] : 0.0f;
        reduce<float>(vals);
        if (threadIdx.x == 0) accumCenters[center * NUM_DIMS * NUM_BLOCKS + dim * NUM_BLOCKS + blockIdx.x] += vals[0];
      }
      __syncthreads();
    }
  }
}

__global__ void kmeansAccumCenters(GPMRGPUConfig gpmrGPUConfig, int * const accumTotals) // a 2d array, stored as [#centers][#blocks]
{
  __shared__ int storage[NUM_BLOCKS];
  int * keySpace = reinterpret_cast<int * >(gpmrGPUConfig.keySpace);
  int * valSpace = reinterpret_cast<int * >(gpmrGPUConfig.valueSpace);

  if (threadIdx.x == 0) keySpace[blockIdx.x * (NUM_DIMS + 1)] = blockIdx.x * (NUM_DIMS + 1);
  storage[threadIdx.x] = accumTotals[blockIdx.x * NUM_BLOCKS + threadIdx.x];
  reduce2<int>(storage);
  if (threadIdx.x == 0) valSpace[blockIdx.x * (NUM_DIMS + 1)] = storage[0];
}
__global__ void kmeansAccumCoords(GPMRGPUConfig gpmrGPUConfig, float * const accumCenters) // a 3d array, stored as [#centers][#dims][#blocks]
{
  __shared__ float storage[NUM_BLOCKS];
  int   * keySpace = reinterpret_cast<int   * >(gpmrGPUConfig.keySpace);
  float * valSpace = reinterpret_cast<float * >(gpmrGPUConfig.valueSpace);

  for (int dim = 0; dim < NUM_DIMS; ++dim)
  {
    if (threadIdx.x == 0) keySpace[blockIdx.x * (NUM_DIMS + 1) + dim + 1] = blockIdx.x * (NUM_DIMS + 1) + dim + 1;
    storage[threadIdx.x] = accumCenters[blockIdx.x * NUM_DIMS * NUM_BLOCKS + dim * NUM_BLOCKS + threadIdx.x];
    reduce2<float>(storage);
    if (threadIdx.x == 0) valSpace[blockIdx.x * (NUM_DIMS + 1) + dim + 1] = storage[0];
    // if (threadIdx.x == 0) valSpace[blockIdx.x * (NUM_DIMS + 1) + dim + 1] = storage[9];
  }
}

void kmeansMapperExecute(const float * const gpuCenters,
                         const int numCenters,
                         const int numDims,
                         void * const gpuMemoryForChunk,
                         const int numElems,
                         float * const accumCenters, // a 2d array, stored as [#centers][#blocks]
                         int   * const accumTotals,  // a 3d array, stored as [#centers][#dims][#blocks]
                         GPMRGPUConfig & gpmrGPUConfig,
                         cudaStream_t & stream)
{
  float * coords = reinterpret_cast<float * >(gpuMemoryForChunk);
  kmeansMapper      <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(gpuCenters, numCenters, coords, numElems, accumCenters, accumTotals);
  kmeansAccumCenters<<<numCenters, NUM_BLOCKS,  0, stream>>>(gpmrGPUConfig, accumTotals);
  kmeansAccumCoords <<<numCenters, NUM_BLOCKS,  0, stream>>>(gpmrGPUConfig, accumCenters);
}
