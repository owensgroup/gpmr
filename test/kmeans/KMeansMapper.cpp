#include <KMeansMapper.h>

#include <gpmr/PreLoadedFixedSizeChunk.h>

#include <cudacpp/Runtime.h>

#include <cstring>
#include <cstdio>

void kmeansMapperExecute(const float * const gpuCenters,
                         const int numCenters,
                         const int numDims,
                         void * const gpuMemoryForChunk,
                         const int numElems,
                         float * const accumCenters,
                         int   * const accumTotals,
                         GPMRGPUConfig & gpmrGPUConfig,
                         cudaStream_t & stream);

KMeansMapper::KMeansMapper(const int pNumCenters, const int pNumDims, const float * const pCenters)
{
  numCenters = pNumCenters;
  numDims = pNumDims;
  centers = new float[numCenters * numDims];
  memcpy(centers, pCenters, numCenters * numDims * sizeof(float));
}
KMeansMapper::~KMeansMapper()
{
  delete [] centers;
}

gpmr::EmitConfiguration KMeansMapper::getEmitConfiguration(gpmr::Chunk * const chunk) const
{
  const int keySize = sizeof(int);
  const int valSize = sizeof(int);
  const int numOutputs = numCenters * (numDims + 1);
  dim3 gridSize(numOutputs, 1, 1);
  dim3 blockSize(1, 1, 1);
  return gpmr::EmitConfiguration::createGridConfiguration(numOutputs * keySize,
                                                          numOutputs * valSize,
                                                          gridSize, blockSize, 1,
                                                          keySize,  valSize);
}
bool KMeansMapper::canExecuteOnGPU() const
{
  return true;
}
bool KMeansMapper::canExecuteOnCPU() const
{
  return false;
}
void KMeansMapper::init()
{
  accumulatedCenters  = reinterpret_cast<float * >(cudacpp::Runtime::malloc(sizeof(float) * numCenters * NUM_BLOCKS * numDims));
  accumulatedTotals   = reinterpret_cast<int   * >(cudacpp::Runtime::malloc(sizeof(int)   * numCenters * NUM_BLOCKS));
  gpuCenters          = reinterpret_cast<float * >(cudacpp::Runtime::malloc(sizeof(float) * numDims * numCenters));
  cudacpp::Runtime::memset(accumulatedCenters, 0, sizeof(float) * numCenters * NUM_BLOCKS * numDims);
  cudacpp::Runtime::memset(accumulatedTotals,  0, sizeof(float) * numCenters * NUM_BLOCKS);
  cudacpp::Runtime::memcpyHtoD(gpuCenters, centers, sizeof(float) * numCenters * numDims);
}
void KMeansMapper::finalize()
{
  cudacpp::Runtime::free(accumulatedCenters);
  cudacpp::Runtime::free(accumulatedTotals);
  cudacpp::Runtime::free(gpuCenters);
}
void KMeansMapper::executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                     cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream)
{
  gpmr::PreLoadedFixedSizeChunk * fsChunk = dynamic_cast<gpmr::PreLoadedFixedSizeChunk * >(chunk);

  kmeansMapperExecute(gpuCenters,
                      numCenters,
                      numDims,
                      gpuMemoryForChunk,
                      fsChunk->getElementCount(),
                      accumulatedCenters,
                      accumulatedTotals,
                      gpmrGPUConfig,
                      kernelStream->getHandle());
}
void KMeansMapper::executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig)
{
}
void KMeansMapper::setCenters(const float * const pCenters)
{
  memcpy(centers, pCenters, sizeof(float) * numCenters * numDims);
}
