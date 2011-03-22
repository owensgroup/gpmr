#include <LinearRegressionReducer.h>

void linearRegressionReducerExecute(const int   * const keys,
                                    const float * const vals,
                                    const int * numVals,
                                    int * const keySpace,
                                    float * const valSpace,
                                    cudaStream_t & stream);

LinearRegressionReducer::LinearRegressionReducer()
{
}
LinearRegressionReducer::~LinearRegressionReducer()
{
}

gpmr::EmitConfiguration LinearRegressionReducer::getEmitConfiguration(const void * const keys,
                                                                      const int * const numVals,
                                                                      const int numKeys,
                                                                      int & numKeysToProcess)
{
  numKeysToProcess = numKeys;
  return gpmr::EmitConfiguration::createGridConfiguration(sizeof(int)    * NUM_CATEGORIES,
                                                          sizeof(float)  * NUM_CATEGORIES,
                                                          dim3(1, 1, 1), dim3(1, 1, 1), NUM_CATEGORIES,
                                                          sizeof(int), sizeof(float));
}
bool LinearRegressionReducer::canExecuteOnGPU() const
{
  return true;
}
bool LinearRegressionReducer::canExecuteOnCPU() const
{
  return true;
}
void LinearRegressionReducer::init()
{
}
void LinearRegressionReducer::finalize()
{
}

void LinearRegressionReducer::executeOnGPUAsync(const int numKeys,
                                                const void * const keys,
                                                const void * const vals,
                                                const int * const keyOffsets,
                                                const int * const valOffsets,
                                                const int * const numVals,
                                                GPMRGPUConfig & gpuConfig,
                                                cudacpp::Stream * const kernelStream)
{
  linearRegressionReducerExecute(reinterpret_cast<const int   * >(keys),
                                 reinterpret_cast<const float * >(vals),
                                 numVals,
                                 reinterpret_cast<int   * >(gpuConfig.keySpace),
                                 reinterpret_cast<float * >(gpuConfig.valueSpace),
                                 kernelStream->getHandle());
}
