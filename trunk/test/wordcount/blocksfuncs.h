#ifndef __BLOCKSFUNCS_H__
#define __BLOCKSFUNCS_H__

const int MAX_BLOCKS        = 1 << 15;
const int THREADS_PER_BLOCK = 256;
const int THREADS_PER_WARP  =  16;
const int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / THREADS_PER_WARP;

template <typename T> struct OpPlus { inline __device__ __host__ T operator () (const T & lhs, const T & rhs) { return lhs + rhs; } };
template <typename T> struct OpMul  { inline __device__ __host__ T operator () (const T & lhs, const T & rhs) { return lhs * rhs; } };

template <typename T, typename Op>
__device__ void reduce(T * const array)
{
  Op op;
  if (threadIdx.x < 128) array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x + 128]);
  __syncthreads();

  if (threadIdx.x < 64) array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x +  64]);
  __syncthreads();

  if (threadIdx.x < 32)
  {
    array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x + 32]);
    array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x + 16]);
    array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x +  8]);
    array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x +  4]);
    array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x +  2]);
    array[threadIdx.x] = op(array[threadIdx.x], array[threadIdx.x +  1]);
  }
  __syncthreads();
}

template <typename T> __device__ void reduceAdd(T * const array) { reduce<T, OpPlus<T> >(array); }
template <typename T> __device__ void reduceMul(T * const array) { reduce<T, OpMul <T> >(array); }

template <typename T, typename Op>
__device__ void inclusiveScan(T * const array)
{
  __shared__ T extra[WARPS_PER_BLOCK];
  Op op;
  T reg = array[threadIdx.x];
  const int THREAD_INDEX_IN_WARP  = (threadIdx.x & 0xF);
  const int WARP_INDEX            = threadIdx.x >> 4;
  if (THREAD_INDEX_IN_WARP > 0) { array[threadIdx.x] = op(array[threadIdx.x - 1], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP > 1) { array[threadIdx.x] = op(array[threadIdx.x - 2], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP > 3) { array[threadIdx.x] = op(array[threadIdx.x - 4], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP > 7) { array[threadIdx.x] = op(array[threadIdx.x - 8], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP == 0) extra[WARP_INDEX] = array[threadIdx.x + THREADS_PER_WARP - 1];
  __syncthreads();
  if (threadIdx.x < WARPS_PER_BLOCK - 1) { extra[threadIdx.x + 1] = op(extra[threadIdx.x], extra[threadIdx.x + 1]); }
  if (threadIdx.x < WARPS_PER_BLOCK - 2) { extra[threadIdx.x + 2] = op(extra[threadIdx.x], extra[threadIdx.x + 2]); }
  if (threadIdx.x < WARPS_PER_BLOCK - 4) { extra[threadIdx.x + 4] = op(extra[threadIdx.x], extra[threadIdx.x + 4]); }
  if (threadIdx.x < WARPS_PER_BLOCK - 8) { extra[threadIdx.x + 8] = op(extra[threadIdx.x], extra[threadIdx.x + 8]); }
  __syncthreads();
  if (WARP_INDEX > 0) array[threadIdx.x] = op(extra[WARP_INDEX - 1], array[threadIdx.x]);
  array[threadIdx.x] -= reg;
}

template <typename T> __device__ void inclusiveScanAdd  (T * const array) { inclusiveScan  <T, OpPlus<T> >(array); }
template <typename T> __device__ void inclusiveScanMul  (T * const array) { inclusiveScan  <T, OpMul <T> >(array); }

template <typename T, typename Op>
__device__ void exclusiveScan(T * const array)
{
  __shared__ T extra[WARPS_PER_BLOCK];
  Op op;
  const int THREAD_INDEX_IN_WARP  = (threadIdx.x & 0xF);
  const int WARP_INDEX            = threadIdx.x >> 4;
  if (THREAD_INDEX_IN_WARP > 0) { array[threadIdx.x] = op(array[threadIdx.x - 1], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP > 1) { array[threadIdx.x] = op(array[threadIdx.x - 2], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP > 3) { array[threadIdx.x] = op(array[threadIdx.x - 4], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP > 7) { array[threadIdx.x] = op(array[threadIdx.x - 8], array[threadIdx.x]); }
  if (THREAD_INDEX_IN_WARP == 0) extra[WARP_INDEX] = array[threadIdx.x + THREADS_PER_WARP - 1];
  __syncthreads();
  if (threadIdx.x < WARPS_PER_BLOCK - 1) { extra[threadIdx.x + 1] = op(extra[threadIdx.x], extra[threadIdx.x + 1]); }
  if (threadIdx.x < WARPS_PER_BLOCK - 2) { extra[threadIdx.x + 2] = op(extra[threadIdx.x], extra[threadIdx.x + 2]); }
  if (threadIdx.x < WARPS_PER_BLOCK - 4) { extra[threadIdx.x + 4] = op(extra[threadIdx.x], extra[threadIdx.x + 4]); }
  if (threadIdx.x < WARPS_PER_BLOCK - 8) { extra[threadIdx.x + 8] = op(extra[threadIdx.x], extra[threadIdx.x + 8]); }
  __syncthreads();
  if (WARP_INDEX > 0) array[threadIdx.x] = op(extra[WARP_INDEX - 1], array[threadIdx.x]);
}

template <typename T> __device__ void exclusiveScanAdd  (T * const array) { exclusiveScan  <T, OpPlus<T> >(array); }
template <typename T> __device__ void exclusiveScanMul  (T * const array) { exclusiveScan  <T, OpMul <T> >(array); }

template <typename T>
__host__ __device__ inline void swap(T & a, T & b)
{
  const T tmp = a;
  a = b;
  b = tmp;
}

template <typename Key, int ELEMENTS_PER_THREAD>
__device__ void bitonicSort(Key * const keys)
{
  __syncthreads();

  for (unsigned int k = 2; k <= THREADS_PER_BLOCK * ELEMENTS_PER_THREAD; k *= 2)
  {
    for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
      const unsigned int tid = threadIdx.x + blockDim.x * i;
      for (unsigned int j = k / 2; j > 0; j /= 2)
      {
        unsigned int ixj = tid ^ j;

        if (ixj > tid)
        {
          if ((tid & k) == 0)
          {
            if (keys[tid] > keys[ixj])
            {
              swap(keys[tid], keys[ixj]);
            }
          }
          else
          {
            if (keys[tid] < keys[ixj])
            {
              swap(keys[tid], keys[ixj]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

template <typename Key>
__device__ void warpBitonicSort(Key * const keys)
{
  const unsigned int tid = threadIdx.x & 0xF;
  #pragma unroll 4
  for (unsigned int k = 0; k < 4; ++k)
  // for (unsigned int k = 2; k <= 16; k *= 2)
  {
    for (unsigned int j = 1 << k; j > 0; j >>= 1)
    // for (unsigned int j = k / 2; j > 0; j /= 2)
    {
      unsigned int ixj = tid ^ j;

      if (ixj > tid)
      {
        if ((tid & k) == 0)
        {
          if (keys[tid] > keys[ixj])
          {
            swap(keys[tid], keys[ixj]);
          }
        }
        else
        {
          if (keys[tid] < keys[ixj])
          {
            swap(keys[tid], keys[ixj]);
          }
        }
      }
    }
  }
}
template <typename Key, typename Value, int ELEMENTS_PER_THREAD>
__device__ void bitonicSort(Key * const keys, Value * const values)
{
  __syncthreads();

  for (unsigned int k = 2; k <= THREADS_PER_BLOCK * ELEMENTS_PER_THREAD; k *= 2)
  {
    for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
      const unsigned int tid = threadIdx.x + blockDim.x * i;
      for (unsigned int j = k / 2; j > 0; j /= 2)
      {
        unsigned int ixj = tid ^ j;

        if (ixj > tid)
        {
          if ((tid & k) == 0)
          {
            if (keys[tid] > keys[ixj])
            {
              swap(keys  [tid], keys  [ixj]);
              swap(values[tid], values[ixj]);
            }
          }
          else
          {
            if (keys[tid] < keys[ixj])
            {
              swap(keys  [tid], keys  [ixj]);
              swap(values[tid], values[ixj]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

template <typename Key, int ELEMENTS_PER_THREAD>
__device__ void compact(Key * const keys, int * const counts)
{
}

#endif
