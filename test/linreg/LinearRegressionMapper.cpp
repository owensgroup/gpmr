#include <LinearRegressionMapper.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <cudacpp/Runtime.h>

void linearRegressionMapperExecute(const void * const points,
                                   const int numPoints,
                                   const int numBlocks,
                                   const int numThreads,
                                   void * const keySpace,
                                   void * const valueSpace,
                                   void * const globalValueSpace,
                                   const bool firstMapping,
                                   cudaStream_t & stream);

LinearRegressionMapper::LinearRegressionMapper()
{
  alreadyMapped = false;
}
LinearRegressionMapper::~LinearRegressionMapper()
{
}

gpmr::EmitConfiguration LinearRegressionMapper::getEmitConfiguration(gpmr::Chunk * const chunk) const
{
  return gpmr::EmitConfiguration::createGridConfiguration(sizeof(int)    * NUM_CATEGORIES,
                                                          sizeof(float)  * NUM_CATEGORIES,
                                                          dim3(1, 1, 1), dim3(1, 1, 1), NUM_CATEGORIES,
                                                          sizeof(int), sizeof(float));
}
bool LinearRegressionMapper::canExecuteOnGPU() const
{
  return true;
}
bool LinearRegressionMapper::canExecuteOnCPU() const
{
  return false;
}
void LinearRegressionMapper::init()
{
  globalVals = cudacpp::Runtime::malloc(sizeof(float) * NUM_CATEGORIES * NUM_BLOCKS);
}
void LinearRegressionMapper::finalize()
{
  cudacpp::Runtime::free(globalVals);
}
void LinearRegressionMapper::executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                               cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream)
{
  gpmr::PreLoadedFixedSizeChunk * fsChunk = dynamic_cast<gpmr::PreLoadedFixedSizeChunk * >(chunk);
  linearRegressionMapperExecute(gpuMemoryForChunk,
                                fsChunk->getElementCount(),
                                NUM_BLOCKS,
                                NUM_THREADS,
                                gpmrGPUConfig.keySpace,
                                gpmrGPUConfig.valueSpace,
                                globalVals,
                                !alreadyMapped,
                                memcpyStream->getHandle());
  alreadyMapped = true;
}
void LinearRegressionMapper::executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig)
{
}
