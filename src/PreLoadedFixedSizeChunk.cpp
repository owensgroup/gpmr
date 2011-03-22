#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <cudacpp/Runtime.h>

namespace gpmr
{
  PreLoadedFixedSizeChunk::PreLoadedFixedSizeChunk(void * const pData,
                                                   const int pElemSize,
                                                   const int pNumElems)
  {
    data = pData;
    elemSize = pElemSize;
    numElems = pNumElems;
  }
  PreLoadedFixedSizeChunk::~PreLoadedFixedSizeChunk()
  {
  }

  void PreLoadedFixedSizeChunk::finishLoading()
  {
  }
  bool PreLoadedFixedSizeChunk::updateQueuePosition(const int newPosition)
  {
    return false;
  }
  int PreLoadedFixedSizeChunk::getMemoryRequirementsOnGPU() const
  {
    return elemSize * numElems;
  }
  void PreLoadedFixedSizeChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
    cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, numElems * elemSize, memcpyStream);
  }
  void PreLoadedFixedSizeChunk::finalizeAsync()
  {
  }
}
