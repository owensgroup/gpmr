#ifndef __GPMR_MAPPER_H__
#define __GPMR_MAPPER_H__

#include <gpmr/EmitConfiguration.h>
#include <gpmr/GPMRCPUConfig.h>
#include <gpmr/GPMRGPUConfig.h>
#include <cudacpp/Stream.h>

namespace gpmr
{
  class Chunk;

  class Mapper
  {
    public:
      Mapper();
      virtual ~Mapper();

      virtual EmitConfiguration getEmitConfiguration(gpmr::Chunk * const chunk) const = 0;
      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                     cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream) = 0;
      virtual void executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig) = 0;
  };
}

#endif
