const int NUM_DIMS = 4;

template <typename T>
__device__ void reduce(const int * const numVals, const void * const oldVals, void * const newVals)
{
  T output = static_cast<T>(0);
  const int count = *numVals;
  const T * input = reinterpret_cast<const T * >(oldVals) + *numVals * blockIdx.x;
  for (int i = 0; i < count; ++i) output += input[i];
  reinterpret_cast<T * >(newVals)[blockIdx.x] = output;
}

__global__ void kmeansReducerKernel(const int numKeys,
                                    const int   * const numVals,
                                    const int   * const oldKeys,
                                          int   * const newKeys,
                                    const void  * const oldVals,
                                          void  * const newVals)
{
  newKeys[blockIdx.x] = oldKeys[blockIdx.x];

  if (blockIdx.x % (NUM_DIMS + 1) == 0) reduce<int  >(numVals, oldVals, newVals);
  else                                  reduce<float>(numVals, oldVals, newVals);
}

void kmeansReducerExecute(const int numKeys,
                          const int   * const numVals,
                          const int   * const oldKeys,
                                int   * const newKeys,
                          const void  * const oldVals,
                                void  * const newVals,
                          cudaStream_t & stream)
{
  kmeansReducerKernel<<<1, numKeys, 0, stream>>>(numKeys, numVals, oldKeys, newKeys, oldVals, newVals);
}
