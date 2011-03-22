#ifndef __HISTOGRAMMAPPER_H__
#define __HISTOGRAMMAPPER_H__

#include <gpmr/Mapper.h>

void intCountMapperExecute(const int numElems,
                            GPMRGPUConfig & gpmrGPUConfig,
                            void * const gpuMemoryForChunk,
                            cudaStream_t & kernelStream);

class IntCountMapper : public gpmr::Mapper
{
  public:
    IntCountMapper();
    virtual ~IntCountMapper();

    virtual gpmr::EmitConfiguration getEmitConfiguration(gpmr::Chunk * const chunk) const;
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();
    virtual void executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                   cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream);
    virtual void executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig);
};

#endif
