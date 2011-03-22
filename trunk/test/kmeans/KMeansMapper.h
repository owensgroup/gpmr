#ifndef __KMEANSMAPPER_H__
#define __KMEANSMAPPER_H__

#include <gpmr/Mapper.h>

class KMeansMapper : public gpmr::Mapper
{
  protected:
    static const int NUM_BLOCKS = 60;
    float * centers;
    int numCenters;
    int numDims;

    float * accumulatedCenters;
    int   * accumulatedTotals;

    float * gpuCenters;
  public:
    KMeansMapper(const int pNumCenters, const int pNumDims, const float * const pCenters);
    virtual ~KMeansMapper();

    virtual gpmr::EmitConfiguration getEmitConfiguration(gpmr::Chunk * const chunk) const;
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();
    virtual void executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                   cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream);
    virtual void executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig);

    void setCenters(const float * const pCenters);

    inline float * getGPUCenters() { return gpuCenters; }
    inline int getNumCenters() const { return numCenters; }
    inline int getNumDims() const { return numDims; }
};

#endif
