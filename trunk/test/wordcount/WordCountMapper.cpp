#include <WordCountMapper.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>

#include <cudacpp/Runtime.h>

#include <mpi.h>

#include <cstring>
#include <cstdio>


void wordCountMapperExecute(const int commRank,
                            const int commSize,
                            const int lineCount,
                            GPMRGPUConfig & config,
                            void * const gpuMemoryForChunk,
                            cudaStream_t & stream,
                            void * const gpuG,
                            const int numUniqueWords,
                            const bool isFirstChunk);

namespace
{
  template <typename T>
  T * readFromFile(const std::string & file, int & numElems)
  {
    T * ret;
    FILE * fp = fopen(file.c_str(), "rb");
    if (fread(&numElems, sizeof(numElems), 1, fp) != 1) printf("Error reading at line %d.\n", __LINE__);
    numElems /= sizeof(T);
    ret = new T[numElems];
    if (fread(ret, numElems * sizeof(T), 1, fp) != 1) printf("Error reading at line %d.\n", __LINE__);
    fclose(fp);
    return ret;
  }
}
WordCountMapper::WordCountMapper(const char * const pDataDir, const int pNumUniqueWords)
{
  int commRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  dataDir = pDataDir;
  numUniqueWords = pNumUniqueWords;
  mappedOneAlready = false;
  int * cpuG;
  unsigned int * cpuT0, * cpuT1, * cpuT2;
  // unsigned int * gpuT0, * gpuT1, * gpuT2;
  int gElems, t0Elems, t1Elems, t2Elems;
  cpuG  = readFromFile<int>         (dataDir + "/wordlist.g",  gElems);
  cpuT0 = readFromFile<unsigned int>(dataDir + "/wordlist.T0", t0Elems);
  cpuT1 = readFromFile<unsigned int>(dataDir + "/wordlist.T1", t1Elems);
  cpuT2 = readFromFile<unsigned int>(dataDir + "/wordlist.T2", t2Elems);
  gpuG  = cudacpp::Runtime::malloc(sizeof(int) * gElems);
  cudacpp::Runtime::memcpyHtoD(gpuG, cpuG, sizeof(int) * gElems);
  cudacpp::Runtime::memcpyToSymbolHtoD("hashConstT0", cpuT0, sizeof(int) * t0Elems, 0);
  cudacpp::Runtime::memcpyToSymbolHtoD("hashConstT1", cpuT1, sizeof(int) * t1Elems, 0);
  cudacpp::Runtime::memcpyToSymbolHtoD("hashConstT2", cpuT2, sizeof(int) * t2Elems, 0);

  delete [] cpuG;
  delete [] cpuT0;
  delete [] cpuT1;
  delete [] cpuT2;
}
WordCountMapper::~WordCountMapper()
{
  cudacpp::Runtime::free(gpuG);
}

gpmr::EmitConfiguration WordCountMapper::getEmitConfiguration(gpmr::Chunk * const chunk) const
{
  const int keySize = sizeof(int);
  const int valSize = sizeof(int);
  dim3 blockSize(256, 1, 1);
  dim3 gridSize((numUniqueWords + blockSize.x - 1) / blockSize.x, 1, 1);
  return gpmr::EmitConfiguration::createGridConfiguration(gridSize.x * blockSize.x * keySize,
                                                          gridSize.x * blockSize.x * valSize,
                                                          gridSize, blockSize, 1,
                                                          keySize,  valSize);
}
bool WordCountMapper::canExecuteOnGPU() const
{
  return true;
}
bool WordCountMapper::canExecuteOnCPU() const
{
  return false;
}
void WordCountMapper::init()
{
}
void WordCountMapper::finalize()
{
}
void WordCountMapper::executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                        cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream)
{
  int commRank, commSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  gpmr::PreLoadedFixedSizeChunk * fsChunk = dynamic_cast<gpmr::PreLoadedFixedSizeChunk * >(chunk);
  const int lineCount = reinterpret_cast<int * >(fsChunk->getData())[1];
  wordCountMapperExecute(commRank,
                         commSize,
                         lineCount,
                         gpmrGPUConfig,
                         gpuMemoryForChunk,
                         kernelStream->getHandle(),
                         gpuG,
                         numUniqueWords,
                         !mappedOneAlready);
  mappedOneAlready = true;
}
void WordCountMapper::executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig)
{
}
