#include <MatMulMapper.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <cudacpp/Runtime.h>

void matMulMapperExecute(const float * const A,
                         const float * const B,
                               float * const C,
                               float * const D,
                         const int matrixFullSize,
                         const int key,
                               int * const keySpace,
                         cudaStream_t & stream);

MatMulMapper::MatMulMapper()
{
  matrixSize = -1;
}
MatMulMapper::~MatMulMapper()
{
}

gpmr::EmitConfiguration MatMulMapper::getEmitConfiguration(gpmr::Chunk * const chunk) const
{
  return gpmr::EmitConfiguration::createGridConfiguration(sizeof(int), sizeof(float) * matrixSize * matrixSize,
                                                          dim3(1, 1, 1), dim3(1, 1, 1), 1,
                                                          sizeof(int), sizeof(float) * matrixSize * matrixSize);
}
bool MatMulMapper::canExecuteOnGPU() const
{
  return true;
}
bool MatMulMapper::canExecuteOnCPU() const
{
  return false;
}
void MatMulMapper::init()
{
  D = reinterpret_cast<float * >(cudacpp::Runtime::malloc(sizeof(float) * matrixSize * 256));
}
void MatMulMapper::finalize()
{
  cudacpp::Runtime::free(D);
}
void MatMulMapper::executeOnGPUAsync(gpmr::Chunk * const chunk, GPMRGPUConfig & gpmrGPUConfig, void * const gpuMemoryForChunk,
                                     cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream)
{
  gpmr::PreLoadedFixedSizeChunk * fsChunk = dynamic_cast<gpmr::PreLoadedFixedSizeChunk * >(chunk);
  int * keyPtr = reinterpret_cast<int * >(fsChunk->getUserData());
  int key = *keyPtr;
  float * A = reinterpret_cast<float * >(gpuMemoryForChunk);
  float * B = A + matrixSize * matrixSize;
  float * C = reinterpret_cast<float * >(gpmrGPUConfig.valueSpace);
  delete keyPtr;
  matMulMapperExecute(A, B, C, D,
                      matrixSize,
                      key,
                      reinterpret_cast<int * >(gpmrGPUConfig.keySpace),
                      memcpyStream->getHandle());
}
void MatMulMapper::executeOnCPUAsync(gpmr::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig)
{
}
