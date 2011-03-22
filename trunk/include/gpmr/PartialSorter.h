#ifndef __GPMR_PARTIALSORTER_H__
#define __GPMR_PARTIALSORTER_H__

#include <gpmr/GPMRCPUConfig.h>
#include <gpmr/GPMRGPUConfig.h>

namespace gpmr
{
  class PartialSorter
  {
    public:
      PartialSorter();
      virtual ~PartialSorter();

      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(GPMRGPUConfig * const gpmrGPUConfig) = 0;
      virtual void executeOnCPUAsync(GPMRCPUConfig * const gpmrCPUConfig) = 0;
  };
}

#endif
