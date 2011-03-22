#ifndef __GPMR_PARTITIONER_H__
#define __GPMR_PARTITIONER_H__

#include <gpmr/GPMRCPUConfig.h>
#include <gpmr/GPMRGPUConfig.h>
#include <cudacpp/Stream.h>

namespace gpmr
{
  class EmitConfiguration;
  class Partitioner
  {
    public:
      Partitioner();
      virtual ~Partitioner();

      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual int  getMemoryRequirementsOnGPU(gpmr::EmitConfiguration & emitConfig) const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(const int numKeys,
                                     const int singleKeySize, const int singleValSize,
                                     void * const gpuKeys,    void * const gpuVals,
                                     int * gpuKeyOffsets,     int * gpuValOffsets,
                                     int * gpuKeyCounts,      int * gpuValCounts,
                                     void * const gpuMemory,
                                     cudacpp::Stream * kernelStream) = 0;
      virtual void executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets) = 0;
  };
}

#endif
