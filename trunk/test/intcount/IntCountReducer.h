#ifndef __HISTOGRAMREDUCER_H__
#define __HISTOGRAMREDUCER_H__

#include <gpmr/Reducer.h>

void intCountReducerExecute(const int numKeys,
                             const int * const keys,
                             const int * const vals,
                             const int * const valOffsets,
                             const int * const numVals,
                             GPMRGPUConfig & gpuConfig,
                             cudaStream_t & stream);

class IntCountReducer : public gpmr::Reducer
{
  protected:
    int * spaceReqsSoFar;
    int lastIndex, lastCount;
    int findNumKeysToScan(const int * const vals, const int start, const int end, const int toSubtract, const int maxElems);
  public:
    IntCountReducer();
    virtual ~IntCountReducer();

    virtual gpmr::EmitConfiguration getEmitConfiguration(const void * const keys,
                                                         const int * const numVals,
                                                         const int numKeys,
                                                         int & numKeysToProcess);
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();

    virtual void executeOnGPUAsync(const int numKeys,
                                   const void * const keys,
                                   const void * const vals,
                                   const int * const keyOffsets,
                                   const int * const valOffsets,
                                   const int * const numVals,
                                   GPMRGPUConfig & gpuConfig,
                                   cudacpp::Stream * const kernelStream);
};

#endif
