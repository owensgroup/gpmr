#include <MatMulMapper.h>
#include <MersenneTwister.h>

#include <gpmr/FixedSizeBinner.h>
#include <gpmr/FixedSizeMapReduceJob.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <gpmr/IntIntSorter.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>

#include <mpi.h>

#include <vector>
#include <cstdlib>
#include <cstdio>

const int MATRIX_SIZE       = 4096;
const int NUM_BUFFERS       =  3;
const int READ_AHEAD        = 10;

int main(int argc, char ** argv)
{
  // testLinearRegressionChunk(argc, argv);
  gpmr::MapReduceJob * job = new gpmr::FixedSizeMapReduceJob(argc, argv, false, true, false);
  cudacpp::Runtime::init();
  cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setQuitOnError(true);
  cudacpp::Runtime::setPrintErrors(true);
  int commRank, commSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  if (argc != 1)
  {
    if (commRank == 0)
    {
      printf("Usage: mpiexec <mpiexec args> %s [optional: num_chunks millions_of_elements].\n", *argv);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete job;
    return 1;
  }

#if 0
  char fileName[1024];
  const int SIZES[] = { 1, 16, 64, 512 };
  const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
  FILE_SIZE = 64;
  void * data = cudacpp::Runtime::mallocHost(FILE_SIZE * sizeof(float2) * 1048576);
  sprintf(fileName, "data/linreg/out_%dmil.dat", FILE_SIZE);

  FILE * fp = fopen(fileName, "rb");
  fread(data, FILE_SIZE * sizeof(float2) * 1048576, 1, fp);
  fclose(fp);

  for (int i = 0; i < NUM_SIZES; ++i)
  {
    double myDataStart  = 1048576.0 * SIZES[i] * static_cast<double>(commRank)      / static_cast<double>(commSize);
    double myDataEnd    = 1048576.0 * SIZES[i] * static_cast<double>(commRank + 1)  / static_cast<double>(commSize);
    myDataEnd = std::min(myDataEnd, SIZES[i] * 1048576.0);
    int myDataSize = static_cast<int>(myDataEnd - myDataStart);
    for (int j = 0; j < 30; ++j)
    {
      int totalDataSize = 0;
      int maxSize = FILE_SIZE * 1048576;
      while (totalDataSize < myDataSize)
      {
        int runSize = std::min(maxSize, myDataSize - totalDataSize);
        job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(float2), runSize));
        totalDataSize += runSize;
      }
      if (commRank == 0) printf("%2d %4d ", commSize, SIZES[i]);
      job->execute();
      fflush(stdout);
    }
  }
  cudacpp::Runtime::freeHost(data);
#else
  MatMulMapper         * mapper         = new MatMulMapper();

  job->setMapper(mapper);

  const int SIZES[] = { 1, 2, 4, 16 };
  const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
  const int matrixSize = 1024;
  for (int sizeIndex = 0; sizeIndex < NUM_SIZES; ++sizeIndex)
  {
    gpmr::Binner * binner = new gpmr::FixedSizeBinner(sizeof(int), sizeof(float) * matrixSize * matrixSize);
    if (job->getBinner() != NULL) delete job->getBinner();
    job->setBinner(binner);

    mapper->setMatrixSize(matrixSize);
    // char fileName[1024];
    void * data = cudacpp::Runtime::mallocHost(matrixSize * matrixSize * sizeof(float) * 2);

    for (int i = 0; i < 3; ++i)
    {
      for (int iIndex = 0; iIndex < SIZES[sizeIndex]; ++iIndex)
      {
        for (int kIndex = 0; kIndex < SIZES[sizeIndex]; ++kIndex)
        {
          for (int jIndex = 0; jIndex < SIZES[sizeIndex]; ++jIndex)
          {
            const int index = iIndex * SIZES[sizeIndex] * SIZES[sizeIndex] + kIndex * SIZES[sizeIndex] + jIndex;
            if (index % commSize == commRank)
            {
              gpmr::PreLoadedFixedSizeChunk * chunk = new gpmr::PreLoadedFixedSizeChunk(data, sizeof(float) * matrixSize * matrixSize, 2);
              int * userData = new int[1];
              *userData = iIndex * SIZES[sizeIndex] + jIndex;
              chunk->setUserData(userData);
              job->addInput(chunk);
            }
          }
        }
      }
      if (commRank == 0) printf("%2d %4d ", commSize, matrixSize * SIZES[sizeIndex]);
      job->execute();
      fflush(stdout);
    }
    cudacpp::Runtime::freeHost(data);
  }
#endif

  delete job;
  cudacpp::Runtime::finalize();

  return 0;
}
