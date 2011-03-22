#ifndef __LINEARREGRESSIONMAPPER_H__
#define __LINEARREGRESSIONMAPPER_H__

#include <gpmr/Mapper.h>

#include <string>

class LinearRegressionMapper : public gpmr::Mapper
{
  protected:
    enum
    {
      NUM_THREADS     = 256,
      NUM_BLOCKS      =  60,
      NUM_CATEGORIES  =   6, // x, y, xx, yy, xy, and numElems
    };
    bool alreadyMapped;
    void * globalVals;
  public:
    LinearRegressionMapper();
    virtual ~LinearRegressionMapper();

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
