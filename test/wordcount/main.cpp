#include <WordCountMapper.h>
#include <WordCountReducer.h>
#include <MersenneTwister.h>

#include <gpmr/FixedSizeBinner.h>
#include <gpmr/PreLoadedFixedSizeChunk.h>
#include <gpmr/FixedSizeMapReduceJob.h>
#include <gpmr/IntIntRoundRobinPartitioner.h>
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
const int NUM_CHUNKS        =     1;
const int NUM_UNIQUE_WORDS  = 42869;

int main(int argc, char ** argv)
{
  // testWordCountChunk(argc, argv);
  gpmr::MapReduceJob * job = new gpmr::FixedSizeMapReduceJob(argc, argv, true);
  cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setQuitOnError(true);
  cudacpp::Runtime::setPrintErrors(true);
  cudacpp::Runtime::init();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (argc > 3)
  {
    if (rank == 0)
    {
      printf("Usage: mpiexec <mpiexec args> %s [optional: max_file_size (Po2)] [optional: num_iters]\n", *argv);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete job;
    return 1;
  }
  int maxFileSize = 0;
  int numIters = 0;

  if (argc > 1) maxFileSize = atoi(argv[1]);
  if (argc > 2) numIters    = atoi(argv[2]);

  if (maxFileSize < 1 || maxFileSize > 512) maxFileSize = 512;
  if (numIters < 1 || numIters > 30) numIters = 30;

  std::vector<gpmr::Chunk * > chunks;
  gpmr::Binner         * binner         = new gpmr::FixedSizeBinner(sizeof(int), sizeof(int));
  gpmr::Sorter         * sorter         = new gpmr::IntIntSorter();
  gpmr::Partitioner    * partitioner    = new gpmr::IntIntRoundRobinPartitioner();
  gpmr::Mapper         * mapper         = new WordCountMapper("data/wordcount", NUM_UNIQUE_WORDS);
  gpmr::Reducer        * reducer        = new WordCountReducer();

  job->setBinner(binner);
  job->setMapper(mapper);
  job->setPartitioner(partitioner);
  job->setSorter(sorter);
  job->setReducer(reducer);

#if 0

  char fileName[1024];
  const int SIZES[] = { 1, 16, 64, 512 };
  const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
  FILE_SIZE = 1;
  void * data = cudacpp::Runtime::mallocHost(FILE_SIZE * sizeof(float2) * 1048576);
  sprintf(fileName, "data/wordcount/out_%dmil.dat", FILE_SIZE);

  FILE * fp = fopen(fileName, "rb");
  fread(data, FILE_SIZE * sizeof(float2) * 1048576, 1, fp);
  fclose(fp);

  for (int i = 0; i < NUM_SIZES; ++i)
  {
    if (i == 1)
    {
      cudacpp::Runtime::freeHost(data);
      FILE_SIZE = 16;
      data = cudacpp::Runtime::mallocHost(FILE_SIZE * sizeof(float2) * 1048576);
      sprintf(fileName, "data/wordcount/out_%dmil.dat", FILE_SIZE);

      FILE * fp = fopen(fileName, "rb");
      fread(data, FILE_SIZE * sizeof(float2) * 1048576, 1, fp);
      fclose(fp);
    }
    int numTimesToReadFile = (SIZES[i] + FILE_SIZE - 1) / FILE_SIZE;

    for (int j = 0; j < 30; ++j)
    {
      for (int k = 0; k < numTimesToReadFile; ++k)
      {
        job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(char), FILE_SIZE * 1048576));
      }
      if (rank == 0) printf("%2d %4d ", size, SIZES[i]);
      job->execute();
      fflush(stdout);
    }
  }
  cudacpp::Runtime::freeHost(data);
#else
  for (FILE_SIZE = 1; FILE_SIZE <= maxFileSize; FILE_SIZE *= 2)
  {
    void * data = cudacpp::Runtime::mallocHost(FILE_SIZE * 1048576);
    char fileName[1024];
    sprintf(fileName, "data/wordcount/out_%dmil.dat", FILE_SIZE);
    FILE * fp = fopen(fileName, "rb");
    if (fread(data, FILE_SIZE * 1048576, 1, fp) != 1)
    {
      printf("Error reading.\n");
      fflush(stdout);
    }
    fclose(fp);

    for (int i = 0; i < numIters; ++i)
    {
      job->addInput(new gpmr::PreLoadedFixedSizeChunk(data, sizeof(char), FILE_SIZE * 1048576));
      if (rank == 0) printf("%5d %5d ", size, FILE_SIZE);
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
