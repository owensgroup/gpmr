#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <gpmr/FixedSizeBinner.h>
#include <gpmr/FixedSizeMapReduceJob.h>
#include <gpmr/IntIntSorter.h>
#include <KMeansMapper.h>
#include <KMeansReducer.h>
#include <MersenneTwister.h>

#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>

const int NUM_DIMS    = 4;
const int NUM_CENTERS = 16;

int main(int argc, char ** argv)
{
  gpmr::MapReduceJob  * job = new gpmr::FixedSizeMapReduceJob(argc, argv, true);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  float * centers = new float[NUM_DIMS * NUM_CENTERS];
  if (rank == 0)
  {
    srand(0);
    for (int i = 0; i < NUM_DIMS * NUM_CENTERS; ++i) centers[i] = static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
  }

  MPI_Bcast(centers, NUM_DIMS * NUM_CENTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);
  job->setBinner (new gpmr::FixedSizeBinner(sizeof(int), sizeof(int)));
  job->setMapper (new KMeansMapper(NUM_CENTERS, NUM_DIMS, centers));
  job->setReducer(new KMeansReducer(NUM_CENTERS, NUM_DIMS));
  job->setSorter (new gpmr::IntIntSorter());

#if 0
  const int SIZES[] = { 1, 8, 32, 512 };
  const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
  int fileSize = 32;
  char name[1024];
  sprintf(name, "data/kmeans/out_%dmil.dat", fileSize);
  void * data = cudacpp::Runtime::mallocHost(fileSize * sizeof(float) * NUM_DIMS * 1048576);
  FILE * fp = fopen(name, "rb");
  fread(data, fileSize * sizeof(float) * NUM_DIMS * 1048576, 1, fp);
  fclose(fp);
  for (int i = 0; i < NUM_SIZES; ++i)
  {
    double myDataStart  = 1048576.0 * SIZES[i] * static_cast<double>(rank)      / static_cast<double>(size);
    double myDataEnd    = 1048576.0 * SIZES[i] * static_cast<double>(rank + 1)  / static_cast<double>(size);
    myDataEnd = std::min(myDataEnd, SIZES[i] * 1048576.0);
    int myDataSize = static_cast<int>(myDataEnd - myDataStart);
    for (int j = 0; j < 30; ++j)
    {
      if (rank == 0) printf("%2d %4d ", size, SIZES[i]);
      int totalDataSize = 0;
      int maxSize = fileSize * 1048576;
      while (totalDataSize < myDataSize)
      {
        int runSize = std::min(maxSize, myDataSize - totalDataSize);
        job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(float) * NUM_DIMS, runSize));
        totalDataSize += runSize;
      }
      job->execute();
      fflush(stdout);
      cudacpp::Runtime::printAllocs(2, 1);
    }
  }
  cudacpp::Runtime::freeHost(data);
#else
  for (int i = 1; i <= 32; i *= 2)
  {
    char name[1024];
    sprintf(name, "data/kmeans/out_%dmil.dat", i);
    void * data = cudacpp::Runtime::mallocHost(i * sizeof(float) * NUM_DIMS * 1048576);
    FILE * fp = fopen(name, "rb");
    if (fread(data, i * sizeof(float) * NUM_DIMS * 1048576, 1, fp) != 1)
    {
      printf("Error reading.\n");
      fflush(stdout);
    }
    fclose(fp);

    for (int j = 0; j < 30; ++j)
    {
      if (rank == 0) printf("%2d %4d ", size, i);
      job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(float) * NUM_DIMS, i * 1048576));
      job->execute();
      fflush(stdout);
      cudacpp::Runtime::printAllocs(2, 1);
    }
    cudacpp::Runtime::freeHost(data);
  }
#endif

  delete [] centers;

  delete job;

  return 0;
}
