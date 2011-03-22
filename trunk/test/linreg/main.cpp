#include <LinearRegressionMapper.h>
#include <LinearRegressionReducer.h>
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

int FILE_SIZE = 1;
const int NUM_BUFFERS       =  3;
const int READ_AHEAD        = 10;

int main(int argc, char ** argv)
{
  // testLinearRegressionChunk(argc, argv);
  gpmr::MapReduceJob * job = new gpmr::FixedSizeMapReduceJob(argc, argv, true);
  cudacpp::Runtime::init();
  cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setQuitOnError(true);
  cudacpp::Runtime::setPrintErrors(true);
  gpmr::Binner         * binner         = new gpmr::FixedSizeBinner(sizeof(int), sizeof(int));
  gpmr::Sorter         * sorter         = new gpmr::IntIntSorter();
  gpmr::Mapper         * mapper         = new LinearRegressionMapper();
  gpmr::Reducer        * reducer        = new LinearRegressionReducer();

  job->setBinner(binner);
  job->setMapper(mapper);
  job->setSorter(sorter);
  job->setReducer(reducer);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (argc != 1)
  {
    if (rank == 0)
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
    double myDataStart  = 1048576.0 * SIZES[i] * static_cast<double>(rank)      / static_cast<double>(size);
    double myDataEnd    = 1048576.0 * SIZES[i] * static_cast<double>(rank + 1)  / static_cast<double>(size);
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
      if (rank == 0) printf("%2d %4d ", size, SIZES[i]);
      job->execute();
      fflush(stdout);
    }
  }
  cudacpp::Runtime::freeHost(data);
#else
  for (FILE_SIZE = 1; FILE_SIZE <= 64; FILE_SIZE *= 2)
  {
    char fileName[1024];
    void * data = cudacpp::Runtime::mallocHost(FILE_SIZE * sizeof(float2) * 1048576);
    sprintf(fileName, "data/linreg/out_%dmil.dat", FILE_SIZE);

    FILE * fp = fopen(fileName, "rb");
    if (fread(data, FILE_SIZE * sizeof(float2) * 1048576, 1, fp) != 1)
    {
      printf("Error reading.\n");
    }
    fclose(fp);

    for (int i = 0; i < 30; ++i)
    {
      job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(float2), FILE_SIZE * 1048576));
      if (rank == 0) printf("%2d %4d ", size, FILE_SIZE);
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
