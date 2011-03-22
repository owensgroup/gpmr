#include <KMeansReducer.h>

#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>

#include <cudpp/cudpp.h>

#include <mpi.h>

void kmeansReducerExecute(const int numKeys,
                          const int   * const numVals,
                          const int   * const oldKeys,
                                int   * const newKeys,
                          const void  * const oldVals,
                                void  * const newVals,
                          cudaStream_t & stream);

KMeansReducer::KMeansReducer(const int pNumCenters, const int pNumDims)
{
  numCenters = pNumCenters;
  numDims = pNumDims;
}
KMeansReducer::~KMeansReducer()
{
}

gpmr::EmitConfiguration KMeansReducer::getEmitConfiguration(const void * const keys,
                                                            const int * const numVals,
                                                            const int numKeys,
                                                            int & numKeysToProcess)
{
  // not used
  numKeysToProcess = numKeys;
  return gpmr::EmitConfiguration::createGridConfiguration(numKeys * sizeof(int), numKeys * sizeof(int),
                                                          dim3(numKeys , 1, 1), dim3(1, 1, 1),
                                                          1,
                                                          sizeof(int), sizeof(int));
}
bool KMeansReducer::canExecuteOnGPU() const
{
  return true;
}
bool KMeansReducer::canExecuteOnCPU() const
{
  return false;
}
void KMeansReducer::init()
{
}
void KMeansReducer::finalize()
{
}

void KMeansReducer::executeOnGPUAsync(const int numKeys,
                                      const void * const keys,
                                      const void * const vals,
                                      const int * keyOffsets,
                                      const int * valOffsets,
                                      const int * numVals,
                                      GPMRGPUConfig & gpuConfig,
                                      cudacpp::Stream * const kernelStream)
{
  kmeansReducerExecute(numKeys,
                       numVals,
                       reinterpret_cast<const int * >(keys),
                       reinterpret_cast<int * >(gpuConfig.keySpace),
                       vals,
                       gpuConfig.valueSpace,
                       kernelStream->getHandle());
}
