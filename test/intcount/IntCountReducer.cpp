#include <IntCountReducer.h>

#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>

#include <cudpp/cudpp.h>

#include <string>

#include <mpi.h>

int IntCountReducer::findNumKeysToScan(const int * const vals, const int start, const int end, const int toSubtract, const int maxElems)
{
  if (start >= end - 1) return start;
  const int mid = (start + end) / 2;
  if      (vals[mid] - toSubtract + mid < maxElems) return findNumKeysToScan(vals, mid,   end,      toSubtract, maxElems);
  else if (vals[mid] - toSubtract + mid > maxElems) return findNumKeysToScan(vals, start, mid - 1,  toSubtract, maxElems);
  return mid;
}

IntCountReducer::IntCountReducer()
{
  spaceReqsSoFar = NULL;
}
IntCountReducer::~IntCountReducer()
{
}

gpmr::EmitConfiguration IntCountReducer::getEmitConfiguration(const void * const keys,
                                                              const int * const numVals,
                                                              const int numKeys,
                                                              int & numKeysToProcess)
{
  const int MAX_OUTPUT_ELEMS = 32 * 1024 * 1024 / (sizeof(int) * 2); // max size is 32MB, each output has a key and value (2 ints).
  if (spaceReqsSoFar == NULL)
  {
    lastIndex = lastCount = 0;
    spaceReqsSoFar = new int[numKeys];
    void * input   = cudacpp::Runtime::malloc(numKeys * sizeof(int));
    void * output  = cudacpp::Runtime::malloc(numKeys * sizeof(int));
    cudacpp::Runtime::memcpyHtoD(input, numVals, sizeof(int) * numKeys);
    CUDPPConfiguration config;
    config.op         = CUDPP_ADD;
    config.datatype   = CUDPP_INT;
    config.algorithm  = CUDPP_SCAN;
    config.options    = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

    CUDPPHandle scanPlan = 0;
    CUDPPResult result   = cudppPlan(&scanPlan, config, numKeys, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
      printf("Error creating CUDPPPlan.\n");
      exit(-1);
    }

    cudppScan(scanPlan, output, input, numKeys);
    cudacpp::Runtime::memcpyDtoH(spaceReqsSoFar, output, numKeys * sizeof(int));

    cudppDestroyPlan(scanPlan);
    cudacpp::Runtime::free(input);
    cudacpp::Runtime::free(output);
  }

  numKeysToProcess = findNumKeysToScan(spaceReqsSoFar + lastIndex, 0, numKeys, lastCount, MAX_OUTPUT_ELEMS) + 1;

  cudacpp::Runtime::sync();
  lastIndex += numKeysToProcess;
  lastCount = spaceReqsSoFar[lastIndex - 1] + numVals[numKeysToProcess - 1];

  dim3 blockSize(1, 1, 1);
  dim3 gridSize(numKeysToProcess, 1, 1);
  return gpmr::EmitConfiguration::createGridConfiguration(numKeysToProcess * sizeof(int),
                                                          numKeysToProcess * sizeof(int),
                                                          gridSize,
                                                          blockSize,
                                                          1,
                                                          sizeof(int),
                                                          sizeof(int));
}
bool IntCountReducer::canExecuteOnGPU() const
{
  return true;
}
bool IntCountReducer::canExecuteOnCPU() const
{
  return false;
}
void IntCountReducer::init()
{
}
void IntCountReducer::finalize()
{
  if (spaceReqsSoFar != NULL)
  {
    delete [] spaceReqsSoFar;
    spaceReqsSoFar = NULL;
  }
}

void IntCountReducer::executeOnGPUAsync(const int numKeys,
                                         const void * const keys,
                                         const void * const vals,
                                         const int * keyOffsets,
                                         const int * valOffsets,
                                         const int * numVals,
                                         GPMRGPUConfig & gpuConfig,
                                         cudacpp::Stream * const kernelStream)
{
  intCountReducerExecute(numKeys,
                         reinterpret_cast<const int * >(keys),
                         reinterpret_cast<const int * >(vals),
                         valOffsets,
                         numVals,
                         gpuConfig,
                         kernelStream->getHandle());
  kernelStream->sync();
}
