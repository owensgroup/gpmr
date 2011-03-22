#ifndef __GPMR_CHUNK_H__
#define __GPMR_CHUNK_H__

#include <cudacpp/Stream.h>

namespace gpmr
{
  class Chunk
  {
    public:
      Chunk();
      virtual ~Chunk();

      virtual void finishLoading() = 0;
      virtual bool updateQueuePosition(const int newPosition) = 0;
      virtual int getMemoryRequirementsOnGPU() const = 0;
      virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream) = 0;
      virtual void finalizeAsync() = 0;
  };
}

#endif
