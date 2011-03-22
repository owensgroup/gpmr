#ifndef __WORDCOUNTMAPPER_H__
#define __WORDCOUNTMAPPER_H__

#include <gpmr/Mapper.h>

#include <string>

void wordCountMapperExecute(const float * const gpuCenters,
                         const int numCenters,
                         const int numDims,
                         void * const gpuMemoryForChunk,
                         const int numElems,
                         GPMRGPUConfig & gpmrGPUConfig,
                         cudaStream_t & stream);

class WordCountMapper : public gpmr::Mapper
{
  protected:
    int numUniqueWords;
    bool mappedOneAlready;
    std::string dataDir;
    void * gpuG;
  public:
    WordCountMapper(const char * const pDataDir, const int numUniqueWords);
    virtual ~WordCountMapper();

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
