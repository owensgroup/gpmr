#include <IntCountMapper.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <cudacpp/Runtime.h>

#include <cstdio>

IntCountMapper::IntCountMapper()
{
}
IntCountMapper::~IntCountMapper()
{
}

gpmr::EmitConfiguration IntCountMapper::getEmitConfiguration(gpmr::Chunk * const chunk) const
{
  gpmr::PreLoadedFixedSizeChunk * fsChunk = static_cast<gpmr::PreLoadedFixedSizeChunk * >(chunk);
  dim3 blockSize(fsChunk->getElementCount(), 1, 1);
  dim3 gridSize(1, 1, 1);
  return gpmr::EmitConfiguration::createGridConfiguration(fsChunk->getElementCount() * sizeof(int),
                                                          fsChunk->getElementCount() * sizeof(int),
                                                          gridSize,     blockSize, 1,
                                                          sizeof(int),  sizeof(int));
}
bool IntCountMapper::canExecuteOnGPU() const
{
  return true;
}
bool IntCountMapper::canExecuteOnCPU() const
{
  return false;
}
void IntCountMapper::init()
{
}
void IntCountMapper::finalize()
{
}
void IntCountMapper::executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                        cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream)
{
  gpmr::PreLoadedFixedSizeChunk * fsChunk = static_cast<gpmr::PreLoadedFixedSizeChunk * >(chunk);
  intCountMapperExecute(fsChunk->getElementCount(), gpmrGPUConfig, gpuMemoryForChunk, kernelStream->getHandle());
}
void IntCountMapper::executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig)
{
}
