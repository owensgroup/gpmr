#ifndef __PRELOADEDFIXEDSIZECHUNK_H__
#define __PRELOADEDFIXEDSIZECHUNK_H__

#include <gpmr/Chunk.h>

namespace gpmr
{
  class PreLoadedFixedSizeChunk : public Chunk
  {
    protected:
      void * data;
      int elemSize, numElems;
      void * userData;
    public:
      PreLoadedFixedSizeChunk(void * const pData,
                              const int pElemSize,
                              const int pNumElems);
      virtual ~PreLoadedFixedSizeChunk();

      virtual void finishLoading();
      virtual bool updateQueuePosition(const int newPosition);
      virtual int getMemoryRequirementsOnGPU() const;
      virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream);
      virtual void finalizeAsync();

      inline void     setUserData(void * const pUserData) { userData = pUserData; }
      inline int      getElementCount() { return numElems;  }
      inline void *   getData()         { return data;      }
      inline void *   getUserData()     { return userData;  }
  };
}

#endif
