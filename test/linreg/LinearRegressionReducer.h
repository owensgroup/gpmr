#ifndef __LINEARREGRESSIONREDUCER_H__
#define __LINEARREGRESSIONREDUCER_H__

#include <gpmr/Reducer.h>
#include <gpmr/EmitConfiguration.h>
#include <gpmr/GPMRGPUConfig.h>

class LinearRegressionReducer : public gpmr::Reducer
{
  protected:
    enum
    {
      NUM_CATEGORIES  =   6, // x, y, xx, yy, xy, and numElems
    };
  public:
    LinearRegressionReducer();
    virtual ~LinearRegressionReducer();

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
