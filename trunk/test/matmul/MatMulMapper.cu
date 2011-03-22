#include <cudacpp/Runtime.h>
#include <cstdio>

__device__ void moveToShared(const float * mat, float subMat[16][16], const int matrixFullSize, const int rowStart, const int colStart)
{
  subMat[threadIdx.x][threadIdx.y] = mat[matrixFullSize * (rowStart + threadIdx.x) + colStart + threadIdx.y];
}

__device__ void addFromShared(float * mat, const float subMat[16][16], const int rowStart, const int colStart)
{
  mat[256 * (rowStart + threadIdx.x) + colStart + threadIdx.y] += subMat[threadIdx.x][threadIdx.y];
}

__device__ void matMul16x16(const float a[16][16],
                            const float b[16][16],
                                  float c[16][16])
{
  for (int i = 0; i < 16; ++i)
  {
    c[threadIdx.x][threadIdx.y] += a[threadIdx.x][i] * b[i][threadIdx.y];
  }
}

__global__ void matMulZeroAndEmit(float * const D, const int key, int * const keySpace)
{
  D[blockIdx.x * blockDim.x + threadIdx.x] = 0.0f;
  // D[gridDim.x * threadIdx.x + blockIdx.x] = 0.0f;
  *keySpace = key;
}

__device__ void matMul256x256(const float * const A,
                              const float * const B,
                                    float * const C,
                              const int matrixFullSize,
                              const int aRowStart,
                              const int aColStart,
                              const int bRowStart,
                              const int bColStart)
{
  __shared__ float subA[16][16];
  __shared__ float subB[16][16];
  __shared__ float subC[16][16];

  subC[threadIdx.y][threadIdx.x] = 0;
  for (int i = 0; i < 16; ++i)
  {
    moveToShared(A, subA, matrixFullSize, aRowStart + i * 16, aColStart);
    for (int j = 0; j < 16; ++j)
    {
      moveToShared(B, subB, matrixFullSize, bRowStart, bColStart + j * 16);
      __syncthreads();
      matMul16x16(subA, subB, subC);
      addFromShared(C, subC, i * 16, j * 16);
    }
  }
}

__global__ void matMulRowXCol(const float * const A,
                              const float * const B,
                                    float * const C,
                              const int matrixFullSize,
                              const int aRowStart,
                              const int bColStart)
{
  matMul256x256(A,
                B,
                C + 256 * 256 * (blockIdx.x * blockDim.x + blockIdx.y),
                matrixFullSize,
                aRowStart,
                256 * blockIdx.x,
                256 * blockIdx.y,
                bColStart);
}

__global__ void matMulCombine(      float * const C,
                              const float * const D,
                              const int matrixFullSize,
                              const int cRowStart,
                              const int cColStart)
{
  __shared__ float sum[16][16];
  const int numSums = matrixFullSize / 256;
  sum[threadIdx.x][threadIdx.y] = 0;
  const int rowSize         = 256;
  const int colSize         = 256;
  const int majorRowOffset  = blockIdx.x * 16;
  const int majorColOffset  = blockIdx.y * 16;
  const int minorRowOffset  = threadIdx.x;
  const int minorColOffset  = threadIdx.y;
  const int rowOffset       = (majorRowOffset + minorRowOffset) * rowSize;
  const int colOffset       = majorColOffset + minorColOffset;
  for (int i = 0; i < numSums; ++i)
  {
    const int matrixOffset = rowSize * colSize * i;
    sum[threadIdx.x][threadIdx.y] += D[matrixOffset + rowOffset + colOffset];
  }
  C[cRowStart * matrixFullSize + rowOffset + colOffset + cColStart] = sum[threadIdx.x][threadIdx.y];
}

__host__ void matMulMapperExecute(const float * const A,
                                  const float * const B,
                                        float * const C,
                                        float * const D,
                                  const int matrixFullSize,
                                  const int key,
                                        int * const keySpace,
                                  cudaStream_t & stream)
{
  dim3 gs(1, 1, 1);
  dim3 bs(16, 16, 1);
  matMulZeroAndEmit<<<matrixFullSize, 256, 0, stream>>>(D, key, keySpace);
  gs.x = gs.y = matrixFullSize / 256;
  for (int i = 0; i < matrixFullSize / 256; ++i)
  {
    for (int j = 0; j < matrixFullSize / 256; ++j)
    {
      matMulRowXCol<<<gs, bs, 0, stream>>>(A, B, D, matrixFullSize, i * 256, j * 256);
      matMulCombine<<<bs, bs, 0, stream>>>(C, D, matrixFullSize, i * 256, j * 256);
    }
  }
}
