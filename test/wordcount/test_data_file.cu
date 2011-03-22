#define USE_CPU 0

#include "blocksfuncs.h"
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA_ERROR()                                                                                        \
{                                                                                                                 \
  cudaError_t err = cudaGetLastError();                                                                           \
  if (err != cudaSuccess)                                                                                         \
  {                                                                                                               \
    printf("%s.%s.%d: %s (error code %d).\n", __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err), err);    \
    fflush(stdout);                                                                                               \
    exit(1);                                                                                                      \
  }                                                                                                               \
}                                                                                                                 \

const int NUM_BLOCKS = 30;

void testSortCPU();
template <typename T> T * readFromFile(const char * const fileName, int & numElems);
void checkOutput(const int numUniqueWords, void * gpuCounts);
void runKernel0(const int numLines,
                void * gpuData,
                void * gpuG,
                void * gpuT0,
                void * gpuT1,
                void * gpuT2,
                void * gpuCounts,
                const int numUniqueWords);
void runKernel1(const int numLines,
                void * gpuData,
                void * gpuG,
                void * gpuCounts,
                const int numUniqueWords);
void runKernel2(void * gpuData,
                void * gpuG,
                void * gpuCounts,
                const int numUniqueWords);
void runKernel3(const int numLines,
                void * gpuData,
                void * gpuG,
                void * gpuCounts,
                const int numUniqueWords);
void runKernel4(void * gpuData,
                void * gpuG,
                const int numUniqueWords,
                void * gpuCounts,
                void * gpuBlockCounts);
void runKernel5(void * gpuData,
                void * gpuG,
                const int numUniqueWords,
                void * gpuCounts,
                void * gpuBlockCounts);


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
__global__ void kernel(int * numWords, int * globalOffset)
{
  *numWords = 0;
  *globalOffset = 0;
}

__global__ void wordCountKernel0(const int blockOffset,
                                 void * const data,
                                 const int * hashG,
                                 const unsigned int * hashT0,
                                 const unsigned int * hashT1,
                                 const unsigned int * hashT2,
                                 unsigned int * const counts)
{
  const int globalIndex = (blockOffset + blockIdx.x) * blockDim.x + threadIdx.x;
  // const int   numWords          = reinterpret_cast<const int * >(data)[0];
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  // const int * const lineLengths = lineOffsets + numLines;
  // int wordCount = 0;
  if (globalIndex < numLines)
  {
    const char * wordData = reinterpret_cast<char * >(data) + lineOffsets[globalIndex];
    while (*wordData != '\n')
    {
      if (*wordData == ' ') ++wordData;
      else
      {
        const int wordIndex = hashFunction(wordData, hashG, hashT0, hashT1, hashT2);
        atomicAdd(counts + wordIndex, 1);
      }
    }
  }
}

__global__ void wordCountKernel1(const int blockOffset,
                                 void * const data,
                                 const int * hashG,
                                 unsigned int * const counts)
{
  const int globalIndex = (blockOffset + blockIdx.x) * blockDim.x + threadIdx.x;
  // const int   numWords          = reinterpret_cast<const int * >(data)[0];
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  // const int * const lineLengths = lineOffsets + numLines;
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

__global__ void wordCountKernel2(void * const data,
                                 const int * hashG,
                                 unsigned int * const counts)
{
  // const int   numWords          = reinterpret_cast<const int * >(data)[0];
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  // const int * const lineLengths = lineOffsets + numLines;
  // int wordCount = 0;
  for (int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; globalIndex < numLines; globalIndex += gridDim.x * blockDim.x)
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

__device__ void checkCharacterForHash(const char key, bool & inWord, unsigned int & f0, unsigned int & f1, unsigned int & f2, int & i, const int * hashG, unsigned int * const counts)
{
  if (key > ' ')
  {
    inWord = true;
    f0 += hashConstT0[i + key];
    f1 += hashConstT1[i + key];
    f2 += hashConstT2[i + key];
    i += 58;
  }
  else if (inWord)
  {
    inWord = false;
    atomicAdd(counts + (hashG[f0 % 52729] + hashG[f1 % 52729] + hashG[f2 % 52729]) % 42869, 1);
    f0 = f1 = f2 = 0;
    i = -65;
  }
}

__global__ void wordCountKernel3(const int blockOffset,
                                 void * const data,
                                 const int * hashG,
                                 unsigned int * const counts)
{
  const int globalIndex = (blockOffset + blockIdx.x) * blockDim.x + threadIdx.x;
  // const int   numWords          = reinterpret_cast<const int * >(data)[0];
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  // int wordCount = 0;
  if (globalIndex < numLines)
  {
  	unsigned int f0 = 0, f1 = 0, f2 = 0;
    int i = -65;
    bool inWord = false;

#if 0
    const char * wordData = reinterpret_cast<const char * >(data) + lineOffsets[globalIndex];
    while (*wordData != '\n')
    {
      checkCharacterForHash(*(wordData++), inWord, f0, f1, f2, i, hashG, counts);
    }
    checkCharacterForHash(*wordData, inWord, f0, f1, f2, i, hashG, counts);
#else
    const char4 * wordData = reinterpret_cast<const char4 * >(reinterpret_cast<const char * >(data) + lineOffsets[globalIndex]);

    char4 patch, nextPatch = *(wordData++);

    do
    {
      patch = nextPatch;
      nextPatch = *(wordData++);
      checkCharacterForHash(patch.x, inWord, f0, f1, f2, i, hashG, counts);
      checkCharacterForHash(patch.y, inWord, f0, f1, f2, i, hashG, counts);
      checkCharacterForHash(patch.z, inWord, f0, f1, f2, i, hashG, counts);
      checkCharacterForHash(patch.w, inWord, f0, f1, f2, i, hashG, counts);
    }
    while (patch.w != '\n');
    checkCharacterForHash(patch.w, inWord, f0, f1, f2, i, hashG, counts);
#endif
  }
}

__global__ void wordCountKernel4(void * const data,
                                 const int * hashG,
                                 const int numUniqueWords,
                                 unsigned int * counts)
{
  __shared__ int wordIndices[THREADS_PER_BLOCK];
  __shared__ int done[THREADS_PER_BLOCK];
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  counts += numUniqueWords * blockIdx.x;

  for (int globalOffset = blockIdx.x * blockDim.x; globalOffset < numLines; globalOffset += gridDim.x * blockDim.x)
  {
    const int globalIndex = globalOffset + threadIdx.x;
    const char * wordData;

    if (globalIndex < numLines) wordData = reinterpret_cast<char * >(data) + lineOffsets[globalIndex];

    do
    {
      if (globalIndex >= numLines || *wordData == '\n')
      {
        wordIndices[threadIdx.x] = -1;
        done[threadIdx.x] = 1;
      }
      else
      {
        while (*wordData == ' ') ++wordData;
        if (*wordData == '\n')
        {
          wordIndices[threadIdx.x] = -1;
          done[threadIdx.x] = 1;
        }
        else
        {
          wordIndices[threadIdx.x] = hashFunction(wordData, hashG, hashConstT0, hashConstT1, hashConstT2);
          done[threadIdx.x] = 0;
        }
      }
#if 0
      bitonicSort<int, 1>(wordIndices);
      const int wordIndex = wordIndices[threadIdx.x];
      int count = 1;
      if (wordIndex != -1 && (threadIdx.x == 0 || wordIndices[threadIdx.x - 1] != wordIndex))
      {
        int index = threadIdx.x + 1;
        while (index < THREADS_PER_BLOCK && wordIndices[index] == wordIndex)
        {
          ++index;
          ++count;
        }
        counts[wordIndex] += count;
      }
#else
      if (wordIndices[threadIdx.x] != -1) counts[wordIndices[threadIdx.x]]++;
#endif
      reduceAdd<int>(done);
    }
    while (done[0] < THREADS_PER_BLOCK);
  }
  __syncthreads();
}

__global__ void wordCountKernel5(void * const data,
                                 const int * hashG,
                                 const int numUniqueWords,
                                 unsigned int * counts)
{
  const int   numLines          = reinterpret_cast<const int * >(data)[1];
  const int * const lineOffsets = reinterpret_cast<const int * >(data) + 32;
  counts += numUniqueWords * blockIdx.x;

  for (int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; globalIndex < numLines; globalIndex += gridDim.x * blockDim.x)
  {
    const char * wordData = reinterpret_cast<char * >(data) + lineOffsets[globalIndex];

    while (*wordData != '\n')
    {
      if (*wordData == ' ') ++wordData;
      else                  counts[hashFunction(wordData, hashG, hashConstT0, hashConstT1, hashConstT2)]++;
    }
  }
}

template <int ELEMS_PER_THREAD>
__global__ void wordCountReduce(const int numUniqueWords, unsigned int * const counts, const unsigned int * const blockCounts)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int accum = 0;
  for (int i = 0; i < ELEMS_PER_THREAD; ++i)
  {
    accum += blockCounts[index + numUniqueWords * i];
  }
  counts[index] = accum;
}

template <int THREADS, int KEYS_PER_THREAD>
__global__ void testSort(int * keys, int * vals)
{
  __shared__ int sharedKeys[THREADS * KEYS_PER_THREAD];
  __shared__ int sharedVals[THREADS * KEYS_PER_THREAD];
  for (int i = 0; i < KEYS_PER_THREAD; ++i)
  {
    sharedKeys[threadIdx.x + blockDim.x * i] = keys[threadIdx.x + blockDim.x * i];
    sharedVals[threadIdx.x + blockDim.x * i] = vals[threadIdx.x + blockDim.x * i];
  }
  bitonicSort<int, int, KEYS_PER_THREAD>(sharedKeys, sharedVals);
  for (int i = 0; i < KEYS_PER_THREAD; ++i)
  {
    keys[threadIdx.x + blockDim.x * i] = sharedKeys[threadIdx.x + blockDim.x * i];
    vals[threadIdx.x + blockDim.x * i] = sharedVals[threadIdx.x + blockDim.x * i];
  }
}

int main(int argc, char ** argv)
{
  const int NUM_KERNELS = 6;
  cudaEvent_t beginEvents[NUM_KERNELS];
  cudaEvent_t endEvents[NUM_KERNELS];
  int numWords, numLines, len, numUniqueWords = 42869;
  char * inputFile, * ptr;

  int gElems, t0Elems, t1Elems, t2Elems;
  int * cpuG;
  unsigned int * cpuT0, * cpuT1, * cpuT2;

  cpuG  = readFromFile<int>         ("/home/stuart/gpmr/data/wordcount/wordlist.g",  gElems);
  cpuT0 = readFromFile<unsigned int>("/home/stuart/gpmr/data/wordcount/wordlist.T0", t0Elems);
  cpuT1 = readFromFile<unsigned int>("/home/stuart/gpmr/data/wordcount/wordlist.T1", t1Elems);
  cpuT2 = readFromFile<unsigned int>("/home/stuart/gpmr/data/wordcount/wordlist.T2", t2Elems);
  FILE * fp = fopen(argv[1], "rb");
  fseek(fp, 0, SEEK_END);
  len = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  inputFile = new char[len];
  fread(inputFile, len, 1, fp);
  fclose(fp);

  ptr = inputFile;
  numWords = *reinterpret_cast<int * >(ptr); ptr += sizeof(int);
  numLines = *reinterpret_cast<int * >(ptr); ptr += sizeof(int);
  ptr += 120;

  const int realNumUniqueWords = numUniqueWords;
  if (numUniqueWords % THREADS_PER_BLOCK != 0) numUniqueWords += THREADS_PER_BLOCK - numUniqueWords % THREADS_PER_BLOCK;

  printf("%d bytes, %d words and %d lines.\n", len, numWords, numLines);
  fflush(stdout);

  void * gpuData, * gpuG, * gpuCounts, * gpuT0, * gpuT1, * gpuT2, * gpuGlobalOffset, * gpuBlockCounts;

  cudaMalloc(&gpuData, len);                                                      CHECK_CUDA_ERROR();
  cudaMalloc(&gpuG,                   gElems * sizeof(int));                      CHECK_CUDA_ERROR();
  cudaMalloc(&gpuT0,                  t0Elems * sizeof(unsigned int));            CHECK_CUDA_ERROR();
  cudaMalloc(&gpuT1,                  t1Elems * sizeof(unsigned int));            CHECK_CUDA_ERROR();
  cudaMalloc(&gpuT2,                  t2Elems * sizeof(unsigned int));            CHECK_CUDA_ERROR();
  cudaMalloc(&gpuCounts,              numUniqueWords * sizeof(int));              CHECK_CUDA_ERROR();
  cudaMalloc(&gpuGlobalOffset,        sizeof(int));                               CHECK_CUDA_ERROR();
  cudaMalloc(&gpuBlockCounts,         numUniqueWords * sizeof(int) * NUM_BLOCKS); CHECK_CUDA_ERROR();

  cudaMemcpy(gpuData, inputFile,  len,                            cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
  cudaMemcpy(gpuG,    cpuG,       gElems  * sizeof(int),          cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
  cudaMemcpy(gpuT0,   cpuT0,      t0Elems * sizeof(unsigned int), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
  cudaMemcpy(gpuT1,   cpuT1,      t1Elems * sizeof(unsigned int), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
  cudaMemcpy(gpuT2,   cpuT2,      t2Elems * sizeof(unsigned int), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();

  cudaMemcpyToSymbol("hashConstT0", cpuT0, t0Elems * sizeof(unsigned int), 0, cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
  cudaMemcpyToSymbol("hashConstT1", cpuT1, t1Elems * sizeof(unsigned int), 0, cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
  cudaMemcpyToSymbol("hashConstT2", cpuT2, t2Elems * sizeof(unsigned int), 0, cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();

  cudaMemset(gpuGlobalOffset, 0, sizeof(int)); CHECK_CUDA_ERROR();

  for (int i = 0; i < NUM_KERNELS; ++i)
  {
    cudaEventCreate(beginEvents + i);
    cudaEventCreate(endEvents   + i);
  }

  delete [] cpuG;
  delete [] cpuT0;
  delete [] cpuT1;
  delete [] cpuT2;
  delete [] inputFile;

  int KERNEL_INDEX = 0;

  cudaMemset(gpuCounts, 0, sizeof(int) * numUniqueWords);                               CHECK_CUDA_ERROR();
  cudaStreamSynchronize(0);                                                             CHECK_CUDA_ERROR();
  cudaEventRecord(beginEvents[KERNEL_INDEX], 0);                                        CHECK_CUDA_ERROR();
  runKernel0(numLines, gpuData, gpuG, gpuT0, gpuT1, gpuT2, gpuCounts, numUniqueWords);  CHECK_CUDA_ERROR();
  cudaEventRecord(endEvents  [KERNEL_INDEX], 0);                                        CHECK_CUDA_ERROR();
  cudaEventSynchronize(endEvents[KERNEL_INDEX]);                                        CHECK_CUDA_ERROR();

  checkOutput(realNumUniqueWords, gpuCounts);
  ++KERNEL_INDEX;

  cudaMemset(gpuCounts, 0, sizeof(int) * numUniqueWords);         CHECK_CUDA_ERROR();
  cudaStreamSynchronize(0);                                       CHECK_CUDA_ERROR();
  cudaEventRecord(beginEvents[KERNEL_INDEX], 0);                  CHECK_CUDA_ERROR();
  runKernel1(numLines, gpuData, gpuG, gpuCounts, numUniqueWords); CHECK_CUDA_ERROR();
  cudaEventRecord(endEvents  [KERNEL_INDEX], 0);                  CHECK_CUDA_ERROR();
  cudaEventSynchronize(endEvents[KERNEL_INDEX]);                  CHECK_CUDA_ERROR();

  checkOutput(realNumUniqueWords, gpuCounts);
  ++KERNEL_INDEX;

  cudaMemset(gpuCounts, 0, sizeof(int) * numUniqueWords); CHECK_CUDA_ERROR();
  cudaStreamSynchronize(0);                               CHECK_CUDA_ERROR();
  cudaEventRecord(beginEvents[KERNEL_INDEX], 0);          CHECK_CUDA_ERROR();
  runKernel2(gpuData, gpuG, gpuCounts, numUniqueWords);   CHECK_CUDA_ERROR();
  cudaEventRecord(endEvents  [KERNEL_INDEX], 0);          CHECK_CUDA_ERROR();
  cudaEventSynchronize(endEvents[KERNEL_INDEX]);          CHECK_CUDA_ERROR();

  checkOutput(realNumUniqueWords, gpuCounts);
  ++KERNEL_INDEX;

  cudaMemset(gpuCounts, 0, sizeof(int) * numUniqueWords);         CHECK_CUDA_ERROR();
  cudaStreamSynchronize(0);                                       CHECK_CUDA_ERROR();
  cudaEventRecord(beginEvents[KERNEL_INDEX], 0);                  CHECK_CUDA_ERROR();
  runKernel3(numLines, gpuData, gpuG, gpuCounts, numUniqueWords); CHECK_CUDA_ERROR();
  cudaEventRecord(endEvents  [KERNEL_INDEX], 0);                  CHECK_CUDA_ERROR();
  cudaEventSynchronize(endEvents[KERNEL_INDEX]);                  CHECK_CUDA_ERROR();

  checkOutput(realNumUniqueWords, gpuCounts);
  ++KERNEL_INDEX;

  cudaMemset(gpuCounts, 0, sizeof(int) * numUniqueWords);                   CHECK_CUDA_ERROR();
  cudaMemset(gpuBlockCounts, 0, sizeof(int) * numUniqueWords * NUM_BLOCKS); CHECK_CUDA_ERROR();
  cudaStreamSynchronize(0);                                                 CHECK_CUDA_ERROR();
  cudaEventRecord(beginEvents[KERNEL_INDEX], 0);                            CHECK_CUDA_ERROR();
  runKernel4(gpuData, gpuG, numUniqueWords, gpuCounts, gpuBlockCounts);     CHECK_CUDA_ERROR();
  cudaEventRecord(endEvents  [KERNEL_INDEX], 0);                            CHECK_CUDA_ERROR();
  cudaEventSynchronize(endEvents[KERNEL_INDEX]);                            CHECK_CUDA_ERROR();

  checkOutput(realNumUniqueWords, gpuCounts);
  ++KERNEL_INDEX;

  cudaMemset(gpuCounts, 0, sizeof(int) * numUniqueWords);                   CHECK_CUDA_ERROR();
  cudaMemset(gpuBlockCounts, 0, sizeof(int) * numUniqueWords * NUM_BLOCKS); CHECK_CUDA_ERROR();
  cudaStreamSynchronize(0);                                                 CHECK_CUDA_ERROR();
  cudaEventRecord(beginEvents[KERNEL_INDEX], 0);                            CHECK_CUDA_ERROR();
  runKernel5(gpuData, gpuG, numUniqueWords, gpuCounts, gpuBlockCounts);     CHECK_CUDA_ERROR();
  cudaEventRecord(endEvents  [KERNEL_INDEX], 0);                            CHECK_CUDA_ERROR();
  cudaEventSynchronize(endEvents[KERNEL_INDEX]);                            CHECK_CUDA_ERROR();

  checkOutput(realNumUniqueWords, gpuCounts);
  ++KERNEL_INDEX;

  for (int i = 0; i < KERNEL_INDEX; ++i)
  {
    float ms;
    cudaEventElapsedTime(&ms, beginEvents[i], endEvents[i]);
    printf("kernel %d took %.3f ms.\n", i, ms);
  }

  return 0;
}

void testSortCPU()
{
  const int THREADS = THREADS_PER_BLOCK;
  const int KEYS_PER_THREAD = 4;
  int * keys = new int[THREADS * KEYS_PER_THREAD];
  int * vals = new int[THREADS * KEYS_PER_THREAD];
  int * gpuKeys, * gpuVals;
  cudaMalloc(reinterpret_cast<void ** >(&gpuKeys), sizeof(int) * THREADS * KEYS_PER_THREAD);
  cudaMalloc(reinterpret_cast<void ** >(&gpuVals), sizeof(int) * THREADS * KEYS_PER_THREAD);

  for (int i = 0; i < THREADS * KEYS_PER_THREAD; ++i)
  {
    keys[i] = i;
    vals[i] = THREADS * KEYS_PER_THREAD - i - 1;
  }
  for (int i = 0; i < THREADS * KEYS_PER_THREAD; ++i)
  {
    const int ind0 = rand() % (THREADS * KEYS_PER_THREAD);
    const int ind1 = rand() % (THREADS * KEYS_PER_THREAD);
    swap(keys[ind0], keys[ind1]);
    swap(vals[ind0], vals[ind1]);
  }
  cudaMemcpy(gpuKeys, keys, sizeof(int) * THREADS * KEYS_PER_THREAD, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuVals, vals, sizeof(int) * THREADS * KEYS_PER_THREAD, cudaMemcpyHostToDevice);
  testSort<THREADS, KEYS_PER_THREAD><<<1, THREADS>>>(gpuKeys, gpuVals);
  cudaMemcpy(keys, gpuKeys, sizeof(int) * THREADS * KEYS_PER_THREAD, cudaMemcpyDeviceToHost);
  cudaMemcpy(vals, gpuVals, sizeof(int) * THREADS * KEYS_PER_THREAD, cudaMemcpyDeviceToHost);

  for (int i = 0; i < THREADS * KEYS_PER_THREAD; ++i)
  {
    printf("%4d: %4d %4d\n", i, keys[i], vals[i]);
  }
  fflush(stdout);

  cudaFree(gpuKeys);
  cudaFree(gpuVals);
  delete [] keys;
  delete [] vals;
}

template <typename T>
T * readFromFile(const char * const fileName, int & numElems)
{
  T * ret;
  FILE * fp = fopen(fileName, "rb");
  fread(&numElems, sizeof(numElems), 1, fp);
  numElems /= sizeof(T);
  ret = new T[numElems];
  fread(ret, numElems * sizeof(T), 1, fp);
  fclose(fp);
  return ret;
}


void runKernel0(const int numLines,
                void * gpuData,
                void * gpuG,
                void * gpuT0,
                void * gpuT1,
                void * gpuT2,
                void * gpuCounts,
                const int numUniqueWords)
{
  const int totalBlocks = (numLines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocksSoFar = 0;
  while (blocksSoFar < totalBlocks)
  {
    int numBlocks = (totalBlocks - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : totalBlocks - blocksSoFar);
    printf("running %d blocks and %d threads with %d blocks before this.\n", numBlocks, THREADS_PER_BLOCK, blocksSoFar);
    wordCountKernel0<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(blocksSoFar,
                                                             gpuData,
                                                             reinterpret_cast<int * >(gpuG),
                                                             reinterpret_cast<unsigned int * >(gpuT0),
                                                             reinterpret_cast<unsigned int * >(gpuT1),
                                                             reinterpret_cast<unsigned int * >(gpuT2),
                                                             reinterpret_cast<unsigned int * >(gpuCounts));
    CHECK_CUDA_ERROR();
    blocksSoFar += numBlocks;
  }
}

void runKernel1(const int numLines,
                void * gpuData,
                void * gpuG,
                void * gpuCounts,
                const int numUniqueWords)
{
  const int totalBlocks = (numLines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocksSoFar = 0;
  while (blocksSoFar < totalBlocks)
  {
    int numBlocks = (totalBlocks - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : totalBlocks - blocksSoFar);
    printf("running %d blocks and %d threads with %d blocks before this.\n", numBlocks, THREADS_PER_BLOCK, blocksSoFar);
    wordCountKernel1<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(blocksSoFar,
                                                             gpuData,
                                                             reinterpret_cast<int * >(gpuG),
                                                             reinterpret_cast<unsigned int * >(gpuCounts));
    CHECK_CUDA_ERROR();
    blocksSoFar += numBlocks;
  }
}

void runKernel2(void * gpuData,
                void * gpuG,
                void * gpuCounts,
                const int numUniqueWords)
{
  int numBlocks = NUM_BLOCKS;
  printf("running %d blocks and %d threads.\n", numBlocks, THREADS_PER_BLOCK);
  wordCountKernel2<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(gpuData,
                                                           reinterpret_cast<int * >(gpuG),
                                                           reinterpret_cast<unsigned int * >(gpuCounts));
  CHECK_CUDA_ERROR();
}

void runKernel3(const int numLines,
                void * gpuData,
                void * gpuG,
                void * gpuCounts,
                const int numUniqueWords)
{
  const int totalBlocks = (numLines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocksSoFar = 0;
  while (blocksSoFar < totalBlocks)
  {
    int numBlocks = (totalBlocks - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : totalBlocks - blocksSoFar);
    printf("running %d blocks and %d threads with %d blocks before this.\n", numBlocks, THREADS_PER_BLOCK, blocksSoFar);
    wordCountKernel3<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(blocksSoFar,
                                                             gpuData,
                                                             reinterpret_cast<int * >(gpuG),
                                                             reinterpret_cast<unsigned int * >(gpuCounts));
    CHECK_CUDA_ERROR();
    blocksSoFar += numBlocks;
  }
}

void runKernel4(void * gpuData,
                void * gpuG,
                int numUniqueWords,
                void * gpuCounts,
                void * gpuBlockCounts)
{
  printf("running %d blocks and %d threads.\n", NUM_BLOCKS, THREADS_PER_BLOCK);
  wordCountKernel4<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(gpuData,
                                                            reinterpret_cast<int * >(gpuG),
                                                            numUniqueWords,
                                                            reinterpret_cast<unsigned int * >(gpuBlockCounts));

  // cudaMemcpyAsync(gpuCounts, gpuBlockCounts, sizeof(int) * numUniqueWords, cudaMemcpyDeviceToDevice, 0);
  int numBlocks = (numUniqueWords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  wordCountReduce<NUM_BLOCKS><<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(numUniqueWords,
                                                                      reinterpret_cast<unsigned int * >(gpuCounts),
                                                                      reinterpret_cast<unsigned int * >(gpuBlockCounts));
  CHECK_CUDA_ERROR();
}

void runKernel5(void * gpuData,
                void * gpuG,
                int numUniqueWords,
                void * gpuCounts,
                void * gpuBlockCounts)
{
  printf("running %d blocks and %d threads.\n", NUM_BLOCKS, THREADS_PER_BLOCK);
  wordCountKernel5<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(gpuData,
                                                            reinterpret_cast<int * >(gpuG),
                                                            numUniqueWords,
                                                            reinterpret_cast<unsigned int * >(gpuBlockCounts));

  // cudaMemcpyAsync(gpuCounts, gpuBlockCounts, sizeof(int) * numUniqueWords, cudaMemcpyDeviceToDevice, 0);
  int numBlocks = (numUniqueWords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  wordCountReduce<NUM_BLOCKS><<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(numUniqueWords,
                                                                      reinterpret_cast<unsigned int * >(gpuCounts),
                                                                      reinterpret_cast<unsigned int * >(gpuBlockCounts));
  CHECK_CUDA_ERROR();
}

void checkOutput(const int numUniqueWords, void * gpuCounts)
{
  int * cpuCounts = new int[numUniqueWords];
  cudaMemcpy(cpuCounts, gpuCounts, sizeof(int) * numUniqueWords,  cudaMemcpyDeviceToHost); CHECK_CUDA_ERROR();

  int zero = 0;
  for (int i = 0; i < numUniqueWords; ++i)
  {
    if (cpuCounts[i] == 0)
    {
      ++zero;
      printf("%d: %d\n", i, cpuCounts[i]);
    }
  }
  for (int i = 0; i < 10; ++i) printf("%d: %d\n", i, cpuCounts[i]);
  printf("%d zero-valued entries.\n", zero);
  delete [] cpuCounts;
}
