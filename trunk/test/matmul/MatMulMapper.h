#ifndef __MATMULMAPPER_H__
#define __MATMULMAPPER_H__

#include <gpmr/Mapper.h>

#include <string>

class MatMulMapper : public gpmr::Mapper
{
  protected:
    bool alreadyMapped;
    float * D;
    int matrixSize;
  public:
    MatMulMapper();
    virtual ~MatMulMapper();

    virtual gpmr::EmitConfiguration getEmitConfiguration(gpmr::Chunk * const chunk) const;
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();
    virtual void executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                   cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream);
    virtual void executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig);

    inline void setMatrixSize(const int pMatrixSize) { matrixSize = pMatrixSize; }
    inline int  getMatrixSize() const { return matrixSize; }
};

#endif
