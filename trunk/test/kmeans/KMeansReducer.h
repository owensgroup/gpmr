#ifndef __KMEANSREDUCER_H__
#define __KMEANSREDUCER_H__

#include <gpmr/Reducer.h>

class KMeansReducer : public gpmr::Reducer
{
  protected:
    int numCenters, numDims;
  public:
    KMeansReducer(const int pNumCenters, const int pNumDims);
    virtual ~KMeansReducer();

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
