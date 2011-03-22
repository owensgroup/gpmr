#include <gpmr/GPMRGPUConfig.h>
#include <cstdio>

const int MAX_BLOCKS        = 1 << 15;
const int THREADS_PER_BLOCK = 256;
// const int THREADS_PER_WARP  =  16;

__constant__ unsigned int hashConstT0[1624];
__constant__ unsigned int hashConstT1[1624];
__constant__ unsigned int hashConstT2[1624];

__host__ __device__ int hashFunction(const char *& key,
                                     const int * const hashG,
                                     const unsigned int * const hashT0,
                                     const unsigned int * const hashT1,
                                     const unsigned int * const hashT2)
{
	unsigned int f0 = 0, f1 = 0, f2 = 0;
  int i = -65;
  while (*key > ' ')
  {
		f0 += hashT0[i + *key];
		f1 += hashT1[i + *key];
		f2 += hashT2[i + *key];
		i += 58;
		++key;
	}

	f0 %= 52729;
	f1 %= 52729;
	f2 %= 52729;

	return (hashG[f0] + hashG[f1] + hashG[f2]) % 42869;
}

__global__ void wordCountMapperInit(const int numUniqueWords, int * const keys, int * const vals)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numUniqueWords; i += gridDim.x * blockDim.x)
  {
    keys[i] = i;
    vals[i] = 0;
  }
}

__global__ void wordCountMapperCountKernel(const int commRank,
                                           const int commSize,
                                           const int blockOffset,
                                           void * const data,
                                           const int * hashG,
                                           unsigned int * const counts)
{
#if 0
  const int globalIndex = ((blockOffset + blockIdx.x) * blockDim.x + threadIdx.x) * commSize + commRank;
#else
  const int globalIndex = ((blockOffset + blockIdx.x) * blockDim.x + threadIdx.x);
#endif
  // const int   numWords          = reinterpret_cast<const int * >(data)[0];
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  //  = lineOffsets + numLines;
  // int wordCount = 0;
  if (globalIndex < numLines)
  {
    const char * wordData = reinterpret_cast<char * >(data) + lineOffsets[globalIndex];
    while (*wordData != '\n')
    {
      if (*wordData == ' ') ++wordData;
      else
      {
        const int wordIndex = hashFunction(wordData, hashG, hashConstT0, hashConstT1, hashConstT2);
        atomicAdd(counts + wordIndex, 1);
      }
    }
  }
}


__host__ void wordCountMapperExecute(const int commRank,
                                     const int commSize,
                                     const int lineCount,
                                     GPMRGPUConfig & config,
                                     void * const gpuMemoryForChunk,
                                     cudaStream_t & stream,
                                     void * const gpuG,
                                     const int numUniqueWords,
                                     const bool isFirstChunk)
{
  if (isFirstChunk)
  {
    int roundedUpWords = (numUniqueWords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
    wordCountMapperInit<<<60, THREADS_PER_BLOCK, 0, stream>>>(roundedUpWords,
                                                              reinterpret_cast<int * >(config.keySpace),
                                                              reinterpret_cast<int * >(config.valueSpace));
  }

  int totalBlocks = (lineCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
#if 0
  totalBlocks /= commSize;
#endif
  int blocksSoFar = 0;
  while (blocksSoFar < totalBlocks)
  {
    int numBlocks = (totalBlocks - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : totalBlocks - blocksSoFar);
    wordCountMapperCountKernel<<<numBlocks, THREADS_PER_BLOCK, 0, stream>>>(blocksSoFar,
                                                                            commRank,
                                                                            commSize,
                                                                            gpuMemoryForChunk,
                                                                            reinterpret_cast<int * >(gpuG),
                                                                            reinterpret_cast<unsigned int * >(config.valueSpace));
    blocksSoFar += numBlocks;
  }
}
