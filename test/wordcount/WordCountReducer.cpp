#include <WordCountReducer.h>

#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>

#include <cudpp/cudpp.h>

#include <mpi.h>

void wordCountReducerExecute(const int numKeys,
                             const void * const keys,
                             const void * const vals,
                             const int * const valOffsets,
                             const int * const numVals,
                             GPMRGPUConfig & gpuConfig,
                             cudaStream_t & stream);

WordCountReducer::WordCountReducer()
{
}
WordCountReducer::~WordCountReducer()
{
}

gpmr::EmitConfiguration WordCountReducer::getEmitConfiguration(const void * const keys,
                                                               const int * const numVals,
                                                               const int numKeys,
                                                               int & numKeysToProcess)
{
  // not used
  numKeysToProcess = numKeys;

  return gpmr::EmitConfiguration::createGridConfiguration(numKeys * sizeof(int),
                                                          numKeys * numVals[0] * sizeof(int),
                                                          dim3((numKeys + NUM_THREADS - 1) / NUM_THREADS, 1, 1),
                                                          dim3(NUM_THREADS, 1, 1),
                                                          1,
                                                          sizeof(int),
                                                          sizeof(int));
}
bool WordCountReducer::canExecuteOnGPU() const
{
  return true;
}
bool WordCountReducer::canExecuteOnCPU() const
{
  return false;
}
void WordCountReducer::init()
{
}
void WordCountReducer::finalize()
{
}

void WordCountReducer::executeOnGPUAsync(const int numKeys,
                                         const void * const keys,
                                         const void * const vals,
                                         const int * keyOffsets,
                                         const int * valOffsets,
                                         const int * numVals,
                                         GPMRGPUConfig & gpuConfig,
                                         cudacpp::Stream * const kernelStream)
{
  wordCountReducerExecute(numKeys, keys, vals, valOffsets, numVals, gpuConfig, kernelStream->getHandle());
}
