#ifndef __GPMR_SORTER_H__
#define __GPMR_SORTER_H__

#include <gpmr/GPMRCPUConfig.h>
#include <gpmr/GPMRGPUConfig.h>

namespace gpmr
{
  class Sorter
  {
    public:
      Sorter();
      virtual ~Sorter();

      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals) = 0;
      virtual void executeOnCPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals) = 0;
  };
}

#endif
