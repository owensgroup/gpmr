#include <gpmr/FixedSizeChunk.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/AsyncFileReader.h>
#include <oscpp/AsyncIORequest.h>
#include <string>
#include <vector>

namespace gpmr
{
  void FixedSizeChunk::loadData(const bool async)
  {
    data = reinterpret_cast<char * >(cudacpp::Runtime::mallocHost(elemSize * numElems));
    reader = new oscpp::AsyncFileReader;
    oscpp::AsyncIORequest * openReq;
    openReq = reader->open(fileName.c_str());
    openReq->sync();
    if (openReq->hasError())
    {
      fprintf(stderr, "Error opening file '%s'.\n", fileName.c_str());
      fflush(stderr);
    }
    delete openReq;

    int soFar = 0;
    while (soFar < numElems)
    {
      int elemsToRead = std::min(numElems - soFar, maxReadSize / elemSize);
      readReqs.push_back(reader->read(data + soFar * elemSize, elemsToRead * elemSize));
      soFar += elemsToRead;
    }

    if (!async) waitForLoad();
  }
  void FixedSizeChunk::waitForLoad()
  {
    if (reader == NULL) return;

    oscpp::AsyncIORequest * closeReq;

    for (unsigned int i = 0; i < readReqs.size(); ++i)
    {
      readReqs[i]->sync();
      if (readReqs[i]->hasError())
      {
        fprintf(stderr, "Error reading from file '%s'.\n", fileName.c_str());
        fflush(stderr);
      }
      delete readReqs[i];
    }
    closeReq = reader->close();
    closeReq->sync();
    if (closeReq->hasError())
    {
      fprintf(stderr, "Error closing file '%s'.\n", fileName.c_str());
      fflush(stderr);
    }
    readReqs.clear();

    delete closeReq;
    delete reader;

    loaded = true;
    reader = NULL;
  }

  FixedSizeChunk::FixedSizeChunk(const std::string & pFileName,
                                 const int pElemSize,
                                 const int pNumElems,
                                 const int pMaxReadSize,
                                 const int readAheadPos)
  {
    loaded = false;
    reader = NULL;
    data = NULL;
    fileName = pFileName;
    elemSize = pElemSize;
    numElems = pNumElems;
    maxReadSize = pMaxReadSize;
    whenToReadAhead = readAheadPos;
    finishLoading();
  }
  FixedSizeChunk::~FixedSizeChunk()
  {
    waitForLoad();
    if (data != NULL) cudacpp::Runtime::freeHost(data);
  }

  void FixedSizeChunk::finishLoading()
  {
    if (loaded) return;
    if (reader == NULL) loadData(false);
    waitForLoad();
  }
  bool FixedSizeChunk::updateQueuePosition(const int newPosition)
  {
    if (newPosition <= whenToReadAhead && reader == NULL)
    {
      loadData(true);
    }
    return true;
  }
  int  FixedSizeChunk::getMemoryRequirementsOnGPU() const
  {
    return numElems * elemSize;
  }
  void FixedSizeChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
    finishLoading();
    cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, numElems * elemSize, memcpyStream);
  }
  void FixedSizeChunk::finalizeAsync()
  {
  }
}
