#ifndef __GPMR_FIXEDSIZERANGEPARTITIONER_H__
#define __GPMR_FIXEDSIZERANGEPARTITIONER_H__

#include <gpmr/Partitioner.h>

namespace gpmr
{
  class EmitConfiguration;

  class FixedSizeRangePartitioner : public Partitioner
  {
    protected:
      int rangeBegin, rangeEnd;
      int commSize;
    public:
      FixedSizeRangePartitioner(const int pRangeBegin, const int pRangeEnd);
      virtual ~FixedSizeRangePartitioner();

      virtual bool canExecuteOnGPU() const;
      virtual bool canExecuteOnCPU() const;
      virtual int  getMemoryRequirementsOnGPU(gpmr::EmitConfiguration & emitConfig) const;
      virtual void init();
      virtual void finalize();
      virtual void executeOnGPUAsync(GPMRGPUConfig gpmrGPUConfig,
                                     int * gpuKeyOffsets, int * gpuValOffsets,
                                     int * gpuKeyCounts,  int * gpuValCounts,
                                     gpmr::EmitConfiguration & emitConfig,
                                     void * const gpuMemory,
                                     cudacpp::Stream * kernelStream);
      virtual void executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets);
  };
}

#include <gpmr/FixedSizeRangerPartitioner.cxx>

#endif
