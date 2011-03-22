#include <IntCountMapper.h>
#include <IntCountReducer.h>
#include <gpmr/IntIntSorter.h>
#include <gpmr/FixedSizeBinner.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <gpmr/IntIntRoundRobinPartitioner.h>
#include <gpmr/IntIntSorter.h>
#include <gpmr/FixedSizeMapReduceJob.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdio>

int main(int argc, char ** argv)
{
  int fileSize;
  // testIntCountChunk(argc, argv);
  gpmr::MapReduceJob  * job         = new gpmr::FixedSizeMapReduceJob(argc, argv);
  gpmr::Binner        * binner      = new gpmr::FixedSizeBinner(sizeof(int), sizeof(int), false);
  gpmr::Mapper        * mapper      = new IntCountMapper;
  gpmr::Partitioner   * partitioner = new gpmr::IntIntRoundRobinPartitioner;
  gpmr::Sorter        * sorter      = new gpmr::IntIntSorter;
  gpmr::Reducer       * reducer     = new IntCountReducer;

#if JDEBUG
  if (argc != 2 || sscanf(argv[1], "%d", &fileSize) != 1 || fileSize < 1 || fileSize > 128 || (fileSize & (fileSize - 1)) != 0)
  {
    printf("argv[1] fileSize fileSize&(fileSize-1) { '%s' %d %08x }\n", argv[1], fileSize, fileSize & (fileSize - 1));
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
      printf("Usage: %s millions_of_elements.\n", *argv);
    }
    MPI_Finalize();
    return 1;
  }
#endif

  // cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setQuitOnError (true);
  cudacpp::Runtime::setPrintErrors (true);
  cudacpp::Runtime::init();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  job->setBinner(binner);
  job->setMapper(mapper);
  job->setPartitioner(partitioner);
  job->setSorter(sorter);
  job->setReducer(reducer);

#if 1
  const int SIZES[] = { 1, 8, 32, 128 };
  const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
  char name[1024];
  fileSize = 32;
  sprintf(name, "data/intcount/out_%dmil.dat", 32);
  void * data = cudacpp::Runtime::mallocHost(fileSize * 1048576 * sizeof(int));
  FILE * fp = fopen(name, "rb");
  if (fread(data, fileSize * 1048576 * sizeof(int), 1, fp)) printf("Error reading.\n");
  fclose(fp);

  for (int i = 0; i < NUM_SIZES; ++i)
  {
    double myDataStart  = 1048576.0 * SIZES[i] * static_cast<double>(rank)      / static_cast<double>(size);
    double myDataEnd    = 1048576.0 * SIZES[i] * static_cast<double>(rank + 1)  / static_cast<double>(size);
    myDataEnd = std::min(myDataEnd, SIZES[i] * 1048576.0);
    int myDataSize = static_cast<int>(myDataEnd - myDataStart);
    for (int j = 0; j < 1; ++j)
    {
      if (rank == 0) printf("%2d %4d ", size, SIZES[i]);
      int totalDataSize = 0;
      int maxSize = fileSize * 1048576;
      while (totalDataSize < myDataSize)
      {
        int runSize = std::min(maxSize, myDataSize - totalDataSize);
        job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(int), runSize));
        totalDataSize += runSize;
      }
      job->execute();
      fflush(stdout);
    }
  }
  cudacpp::Runtime::freeHost(data);
#else
  for (fileSize = 1; fileSize <= 32; fileSize *= 2)
  {
    char name[1024];
    sprintf(name, "data/intcount/out_%dmil.dat", fileSize);
    void * data = cudacpp::Runtime::mallocHost(fileSize * 1048576 * sizeof(int));
    FILE * fp = fopen(name, "rb");
    fread(data, fileSize * 1048576 * sizeof(int), 1, fp);
    fclose(fp);

    for (int j = 0; j < 30; ++j)
    {
      if (rank == 0) printf("%2d %4d ", size, fileSize);
      job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(int), fileSize * 1048576));
      job->execute();
      fflush(stdout);
      // cudacpp::Runtime::printAllocs(2, 1);
    }
    cudacpp::Runtime::freeHost(data);
  }
#endif

  delete job;

  return 0;
}
