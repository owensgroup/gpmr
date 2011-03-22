#include <gpmr/FixedSizeMapReduceJob.h>
#include <gpmr/Combiner.h>
#include <gpmr/Chunk.h>
#include <gpmr/EmitConfiguration.h>
#include <gpmr/GPMRCPUConfig.h>
#include <gpmr/GPMRGPUConfig.h>
#include <gpmr/Mapper.h>
#include <gpmr/PartialReducer.h>
#include <gpmr/Partitioner.h>
#include <gpmr/Reducer.h>
#include <gpmr/Sorter.h>
#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <mpi.h>

#include <algorithm>
#include <vector>
#include <cstring>
#include <string>

namespace gpmr
{

  void FixedSizeMapReduceJob::determineMaximumSpaceRequirements()
  {
    const int combinerMem = combiner == NULL ? 0 : combiner->getMemoryRequirementsOnGPU();
    maxStaticMem = std::max(maxStaticMem, combinerMem);
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      EmitConfiguration emitConfig = mapper->getEmitConfiguration(chunks[i]);
      emitConfigs.push_back(emitConfig);
      const int chunkMem    = chunks[i]->getMemoryRequirementsOnGPU();
      const int partMem     = partitioner     == NULL ? 0 : partitioner->getMemoryRequirementsOnGPU(emitConfig);
      const int partialMem  = partialReducer  == NULL ? 0 : partialReducer->getMemoryRequirementsOnGPU(emitConfig);
      maxStaticMem  = std::max(maxStaticMem,  chunkMem + std::max(partMem, partialMem));
      maxKeySpace   = std::max(maxKeySpace,  emitConfig.getKeySpace());
      maxValSpace   = std::max(maxValSpace,  emitConfig.getValueSpace());
    }
    maxStaticMem = std::max(maxStaticMem, 1);
    maxKeySpace  = std::max(maxKeySpace,  1);
    maxValSpace  = std::max(maxValSpace,  1);
  }
  void FixedSizeMapReduceJob::allocateMapVariables()
  {
    gpuKeys       = cudacpp::Runtime::malloc    (numBuffers * maxKeySpace);
    gpuVals       = cudacpp::Runtime::malloc    (numBuffers * maxValSpace);
    gpuStaticMems = cudacpp::Runtime::malloc    (numBuffers * maxStaticMem);
    cpuKeys       = cudacpp::Runtime::mallocHost(numBuffers * maxKeySpace);
    cpuVals       = cudacpp::Runtime::mallocHost(numBuffers * maxValSpace);

    gpuKeyOffsets = reinterpret_cast<int * >(cudacpp::Runtime::malloc    (numBuffers * commSize * sizeof(int) * 4));
    gpuValOffsets = gpuKeyOffsets + commSize * 1;
    gpuKeyCounts  = gpuKeyOffsets + commSize * 2;
    gpuValCounts  = gpuKeyOffsets + commSize * 3;
    cpuKeyOffsets = reinterpret_cast<int * >(cudacpp::Runtime::mallocHost(numBuffers * commSize * sizeof(int) * 4));
    cpuValOffsets = cpuKeyOffsets + commSize * 1;
    cpuKeyCounts  = cpuKeyOffsets + commSize * 2;
    cpuValCounts  = cpuKeyOffsets + commSize * 3;
  }
  void FixedSizeMapReduceJob::freeMapVariables()
  {
    cudacpp::Runtime::free    (gpuKeys);
    cudacpp::Runtime::free    (gpuVals);
    cudacpp::Runtime::free    (gpuStaticMems);
    cudacpp::Runtime::freeHost(cpuKeys);
    cudacpp::Runtime::freeHost(cpuVals);
    cudacpp::Runtime::free    (gpuKeyOffsets);
    cudacpp::Runtime::freeHost(cpuKeyOffsets);
  }
  void FixedSizeMapReduceJob::startBinnerThread()
  {
    binnerThread = new oscpp::Thread(binner);
    binnerThread->start();
  }
  void FixedSizeMapReduceJob::mapChunkExecute(const unsigned int chunkIndex,
                                              GPMRGPUConfig & config,
                                              void * const memPool)
  {
    chunks[chunkIndex]->stageAsync(memPool, kernelStream);
    mapper->executeOnGPUAsync(chunks[chunkIndex], config, memPool, kernelStream, memcpyStream);
    if (partialReducer != NULL) partialReducer->executeOnGPUAsync(emitConfigs[chunkIndex], config.keySpace, config.valueSpace, memPool, gpuKeyCounts, gpuKeyOffsets, gpuValCounts, gpuValOffsets, kernelStream);
  }
  void FixedSizeMapReduceJob::mapChunkMemcpy(const unsigned int chunkIndex,
                                             const void * const gpuKeySpace,
                                             const void * const gpuValueSpace)
  {
    if (!accumMap || chunkIndex + 1 == chunks.size())
    {
      cudacpp::Runtime::memcpyDtoHAsync(cpuKeys,        gpuKeySpace,   maxKeySpace,             memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuVals,        gpuValueSpace, maxValSpace,             memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuKeyOffsets,  gpuKeyOffsets, commSize * sizeof(int),  memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuValOffsets,  gpuValOffsets, commSize * sizeof(int),  memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuKeyCounts,   gpuKeyCounts,  commSize * sizeof(int),  memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuValCounts,   gpuValCounts,  commSize * sizeof(int),  memcpyStream);
    }
  }
  void FixedSizeMapReduceJob::mapChunkPartition(const unsigned int chunkIndex,
                                                void * const memPool,
                                                GPMRGPUConfig & config)
  {
    if (combiner == NULL && accumMap && chunkIndex == chunks.size() - 1)
    {
      memcpyStream->sync();
      partitionSub(memPool,
                   config.keySpace,
                   config.valueSpace,
                   emitConfigs[chunkIndex].getIndexCount(),
                   emitConfigs[chunkIndex].getKeySize(),
                   emitConfigs[chunkIndex].getValueSize());
    }
    if (combiner == NULL && !accumMap) partitionChunk(chunkIndex);
  }

  void FixedSizeMapReduceJob::queueChunk(const unsigned int chunkIndex)
  {
    GPMRGPUConfig config;
    void * memPool    = reinterpret_cast<char * >(gpuStaticMems) + (maxStaticMem * chunkIndex) % numBuffers;
    config.keySpace   = reinterpret_cast<char * >(gpuKeys) + (maxKeySpace * chunkIndex) % numBuffers;
    config.valueSpace = reinterpret_cast<char * >(gpuVals) + (maxValSpace * chunkIndex) % numBuffers;
    config.emitInfo.grid.numThreads     = emitConfigs[chunkIndex].getThreadCount();
    config.emitInfo.grid.emitsPerThread = emitConfigs[chunkIndex].getEmitsPerThread();

    mapChunkExecute(chunkIndex, config, memPool);
    mapChunkMemcpy(chunkIndex, config.keySpace, config.valueSpace);
    if (reducer != NULL) mapChunkPartition(chunkIndex, memPool, config);
  }
  void FixedSizeMapReduceJob::partitionSubDoGPU(void * const memPool,
                                                void * const keySpace,
                                                void * const valueSpace,
                                                const int numKeys,
                                                const int singleKeySize,
                                                const int singleValSize)
  {
    partitioner->executeOnGPUAsync(numKeys,
                                   singleKeySize, singleValSize,
                                   keySpace,      valueSpace,
                                   gpuKeyOffsets, gpuValOffsets,
                                   gpuKeyCounts,  gpuValCounts,
                                   memPool,
                                   kernelStream);
    cudacpp::Runtime::memcpyDtoH(cpuKeyCounts,  gpuKeyCounts,  sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuValCounts,  gpuValCounts,  sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuKeyOffsets, gpuKeyOffsets, sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuValOffsets, gpuValOffsets, sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuKeys, keySpace,   singleKeySize * numKeys);
    cudacpp::Runtime::memcpyDtoH(cpuVals, valueSpace, singleValSize * numKeys);
  }
  void FixedSizeMapReduceJob::partitionSubDoNullPartitioner(const int numKeys)
  {
    cpuKeyOffsets[0] = cpuValOffsets[0] = 0;
    cpuKeyCounts[0] = cpuValCounts[0] = numKeys;
    for (int index = 1; index < commSize; ++index)
    {
      cpuKeyOffsets[index] = cpuValOffsets[index] = 0;
      cpuKeyCounts [index] = cpuValCounts [index] = 0;
    }
  }
  void FixedSizeMapReduceJob::partitionSubSendData(const int singleKeySize, const int singleValSize)
  {
    int totalKeySize = 0;
    for (int index = 0; index < commSize; ++index)
    {
      int i = (index + commRank) % commSize;
      int keyBytes = cpuKeyCounts[i] * singleKeySize;
      int valBytes = cpuValCounts[i] * singleValSize;
      totalKeySize += keyBytes;

      if (keyBytes + valBytes > 0) // it can happen that we send nothing.
      {
        oscpp::AsyncIORequest * ioReq = binner->sendTo(i,
                                                       reinterpret_cast<char * >(cpuKeys) + cpuKeyOffsets[i] * singleKeySize,
                                                       reinterpret_cast<char * >(cpuVals) + cpuValOffsets[i] * singleValSize,
                                                       keyBytes,
                                                       valBytes);
        sendReqs.push_back(ioReq);
      }
    }
  }
  void FixedSizeMapReduceJob::partitionCheckSends(const bool sync)
  {
    std::vector<oscpp::AsyncIORequest * > newReqs;
    for (unsigned int j = 0; j < sendReqs.size(); ++j)
    {
      if (sync) sendReqs[j]->sync();
      if (sendReqs[j]->query()) delete sendReqs[j];
      else                      newReqs.push_back(sendReqs[j]);
    }
    sendReqs = newReqs;
  }
  void FixedSizeMapReduceJob::partitionSub(void * const memPool,
                                           void * const keySpace,
                                           void * const valueSpace,
                                           const int numKeys,
                                           const int singleKeySize,
                                           const int singleValSize)
  {
    partitionCheckSends(false);
    if (partitioner != NULL)
    {
      partitionSubDoGPU(memPool, keySpace, valueSpace, numKeys, singleKeySize, singleValSize);
    }
    if (partitioner == NULL) // everything goes to rank 0
    {
      partitionSubDoNullPartitioner(numKeys);
    }
    partitionSubSendData(singleKeySize, singleValSize);
    if (syncPartSends) partitionCheckSends(true);
  }
  void FixedSizeMapReduceJob::partitionChunk(const unsigned int chunkIndex)
  {
    void * memPool    = reinterpret_cast<char * >(gpuStaticMems) + (maxStaticMem * chunkIndex) % numBuffers;
    void * keySpace   = reinterpret_cast<char * >(gpuKeys) + (maxKeySpace * chunkIndex) % numBuffers;
    void * valueSpace = reinterpret_cast<char * >(gpuVals) + (maxValSpace * chunkIndex) % numBuffers;
    partitionSub(memPool,
                 keySpace,
                 valueSpace,
                 emitConfigs[chunkIndex].getIndexCount(),
                 emitConfigs[chunkIndex].getKeySize(),
                 emitConfigs[chunkIndex].getValueSize());
  }
  void FixedSizeMapReduceJob::saveChunk(const unsigned int chunkIndex)
  {
    const int numKeys = emitConfigs[chunkIndex].getIndexCount();
    void * keySpace = reinterpret_cast<char * >(cpuKeys) + (maxKeySpace * chunkIndex) % numBuffers;
    void * valSpace = reinterpret_cast<char * >(cpuVals) + (maxValSpace * chunkIndex) % numBuffers;
    void * keysToSave = new char[numKeys * emitConfigs[chunkIndex].getKeySize()];
    void * valsToSave = new char[numKeys * emitConfigs[chunkIndex].getValueSize()];
    memcpy(keysToSave, keySpace, numKeys * emitConfigs[chunkIndex].getKeySize());
    memcpy(valsToSave, valSpace, numKeys * emitConfigs[chunkIndex].getValueSize());
    savedKeys.push_back(keySpace);
    savedVals.push_back(valSpace);
    keyAndValCount.push_back(numKeys);
  }
  void FixedSizeMapReduceJob::combine()
  {
  }
  void FixedSizeMapReduceJob::globalPartition()
  {
    const int singleKeySize = emitConfigs[0].getKeySize();
    const int singleValSize = emitConfigs[0].getValueSize();
    int numKeys = 0, totalKeySize, totalValSize;
    for (unsigned int i = 0; i < keyAndValCount.size(); ++i)
    {
      numKeys += keyAndValCount[i];
    }
    totalKeySize = numKeys * singleKeySize;
    totalValSize = numKeys * singleValSize;
    char * gpuFullKeys = reinterpret_cast<char * >(cudacpp::Runtime::malloc(totalKeySize));
    char * gpuFullVals = reinterpret_cast<char * >(cudacpp::Runtime::malloc(totalValSize));

    int offset = 0;
    for (unsigned int i = 0; i < keyAndValCount.size(); ++i)
    {
      cudacpp::Runtime::memcpyHtoD(gpuFullKeys + offset, savedKeys[i], keyAndValCount[i] * singleKeySize);
      cudacpp::Runtime::memcpyHtoD(gpuFullVals + offset, savedVals[i], keyAndValCount[i] * singleValSize);
    }
    partitionSub(gpuStaticMems, gpuFullKeys, gpuFullVals, numKeys, singleKeySize, singleValSize);
    cudacpp::Runtime::free(gpuFullKeys);
    cudacpp::Runtime::free(gpuFullVals);
  }
  void FixedSizeMapReduceJob::enqueueAllChunks()
  {
    if (reducer == NULL) binner->finish();
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      queueChunk(i);
      if (combiner != NULL && !accumMap) saveChunk(i);
    }
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      delete chunks[i];
    }
    if (combiner != NULL)
    {
      combine();
    }
    if (combiner != NULL)
    {
      globalPartition();
    }
    chunks.clear();
  }
  void FixedSizeMapReduceJob::collectVariablesFromBinnerAndKill()
  {
    if (reducer != NULL) binner->finish();
    binnerThread->join();
    delete binnerThread;
    if (sorter != NULL || reducer != NULL)
    {
      binner->getFinalDataSize(keySize, valSize);
      if (keySize > 0)
      {
        keys = cudacpp::Runtime::mallocHost(keySize);
        vals = cudacpp::Runtime::mallocHost(valSize);
        binner->getFinalData(keys, vals);
      }
    }
  }
  void FixedSizeMapReduceJob::getReduceRunParameters()
  {
    maxInputKeySpace = maxInputValSpace = maxInputValOffsetSpace = maxInputNumValsSpace = 0;

    int keysSoFar = 0;
    while (keysSoFar < numUniqueKeys)
    {
      char * tKeys = reinterpret_cast<char * >(keys) + keysSoFar * keySize;
      int numKeysToProcess;
      EmitConfiguration emitConfig = reducer->getEmitConfiguration(tKeys, numVals + keysSoFar, numUniqueKeys - keysSoFar, numKeysToProcess);
      maxKeySpace             = std::max(maxKeySpace,             emitConfig.getKeySpace());
      maxValSpace             = std::max(maxValSpace,             emitConfig.getValueSpace());
      maxInputKeySpace        = std::max(maxInputKeySpace,        numKeysToProcess * static_cast<int>(sizeof(int)));
      maxInputValSpace        = std::max(maxInputValSpace,        (valOffsets[keysSoFar + numKeysToProcess - 1] - valOffsets[keysSoFar] + numVals[keysSoFar + numKeysToProcess - 1]) * static_cast<int>(sizeof(int)));
      maxInputValOffsetSpace  = std::max(maxInputValOffsetSpace,  numKeysToProcess * static_cast<int>(sizeof(int)));
      maxInputNumValsSpace    = std::max(maxInputNumValsSpace,    numKeysToProcess * static_cast<int>(sizeof(int)));
      keysSoFar += numKeysToProcess;
      emitConfigs.push_back(emitConfig);
      keyCount.push_back(numKeysToProcess);
    }
  }
  GPMRGPUConfig & FixedSizeMapReduceJob::getReduceConfig(const int index)
  {
    return configs[index % configs.size()];
  }
  void FixedSizeMapReduceJob::allocateReduceVariables()
  {
    gpuKeys             = cudacpp::Runtime::malloc    (numBuffers * maxKeySpace);
    gpuVals             = cudacpp::Runtime::malloc    (numBuffers * maxValSpace);
    gpuInputKeys        = cudacpp::Runtime::malloc    (numBuffers * maxInputKeySpace);
    gpuInputVals        = cudacpp::Runtime::malloc    (numBuffers * maxInputValSpace);
    gpuInputValOffsets  = reinterpret_cast<int * >(cudacpp::Runtime::malloc(numBuffers * maxInputValOffsetSpace));
    gpuInputValCounts   = reinterpret_cast<int * >(cudacpp::Runtime::malloc(numBuffers * maxInputNumValsSpace));
    cpuKeys             = cudacpp::Runtime::mallocHost(numBuffers * maxKeySpace);
    cpuVals             = cudacpp::Runtime::mallocHost(numBuffers * maxValSpace);
  }
  void FixedSizeMapReduceJob::freeReduceVariables()
  {
    if (keys               != NULL) cudacpp::Runtime::freeHost(keys);
    if (vals               != NULL) cudacpp::Runtime::freeHost(vals);
    if (gpuKeys            != NULL) cudacpp::Runtime::free    (gpuKeys);
    if (gpuVals            != NULL) cudacpp::Runtime::free    (gpuVals);
    if (gpuInputKeys       != NULL) cudacpp::Runtime::free    (gpuInputKeys);
    if (gpuInputVals       != NULL) cudacpp::Runtime::free    (gpuInputVals);
    if (gpuInputValOffsets != NULL) cudacpp::Runtime::free    (gpuInputValOffsets);
    if (gpuInputValCounts  != NULL) cudacpp::Runtime::free    (gpuInputValCounts);
    if (cpuKeys            != NULL) cudacpp::Runtime::freeHost(cpuKeys);
    if (cpuVals            != NULL) cudacpp::Runtime::freeHost(cpuVals);
    if (valOffsets         != NULL) cudacpp::Runtime::freeHost(valOffsets);
    if (numVals            != NULL) cudacpp::Runtime::freeHost(numVals);
  }
  void FixedSizeMapReduceJob::copyReduceInput(const int index, const int keysSoFar)
  {
    const int keyCount = std::min(numUniqueKeys, emitConfigs[index].getIndexCount());
    const int keyBytesToCopy  = keyCount * sizeof(int);
    const int valBytesToCopy  = (valOffsets[keysSoFar + keyCount - 1] - valOffsets[keysSoFar] + numVals[keysSoFar + keyCount - 1]) * sizeof(int);

    int numValsToProcess = 0;
    for (int i = 0; i < keyCount; ++i)
    {
      numValsToProcess += numVals[keysSoFar + i];
    }

    cudacpp::Runtime::memcpyHtoDAsync(gpuInputKeys, reinterpret_cast<int * >(keys) + keysSoFar,             keyBytesToCopy, memcpyStream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuInputVals, reinterpret_cast<int * >(vals) + valOffsets[keysSoFar], valBytesToCopy, memcpyStream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuInputValOffsets, valOffsets + keysSoFar, keyCount * sizeof(int), memcpyStream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuInputValCounts,  numVals    + keysSoFar, keyCount * sizeof(int), memcpyStream);
  }
  void FixedSizeMapReduceJob::executeReduce(const int index)
  {
    reducer->executeOnGPUAsync(keyCount[index],
                               gpuInputKeys,
                               gpuInputVals,
                               NULL,
                               gpuInputValOffsets,
                               gpuInputValCounts,
                               getReduceConfig(index),
                               kernelStream);
  }
  void FixedSizeMapReduceJob::copyReduceOutput(const int index, const int keysSoFar)
  {
    cudacpp::Runtime::memcpyDtoHAsync(reinterpret_cast<int * >(cpuKeys) + keysSoFar, getReduceConfig(index).keySpace,   keyCount[index] * sizeof(int), memcpyStream);
    cudacpp::Runtime::memcpyDtoHAsync(reinterpret_cast<int * >(cpuVals) + keysSoFar, getReduceConfig(index).valueSpace, keyCount[index] * sizeof(int), memcpyStream);
  }
  void FixedSizeMapReduceJob::enqueueReductions()
  {
    int keysSoFar = 0;
    for (unsigned int i = 0; i < emitConfigs.size(); ++i)
    {
      EmitConfiguration & emitConfig        = emitConfigs[i];
      getReduceConfig(i).emitInfo.grid.numThreads       = emitConfig.getThreadCount();
      getReduceConfig(i).emitInfo.grid.emitsPerThread   = emitConfig.getEmitsPerThread();

      copyReduceInput(i, keysSoFar);
      cudacpp::Runtime::sync();
      executeReduce(i);
      cudacpp::Runtime::sync();
      copyReduceOutput(i, keysSoFar);
      cudacpp::Runtime::sync();

      cudacpp::Stream::nullStream.sync();

      keysSoFar += keyCount[i];
    }
  }
  void FixedSizeMapReduceJob::map()
  {
    emitConfigs.clear();
    savedKeys.clear();
    savedVals.clear();
    keyAndValCount.clear();
    cudacpp::DeviceProperties * props = cudacpp::DeviceProperties::get(getDeviceNumber());
    kernelStream = memcpyStream = &cudacpp::Stream::nullStream;

    maxStaticMem  = 0;
    maxKeySpace = 0;
    maxValSpace = 0;

    // int maxCountSpace = commSize * sizeof(int) * 2;
    // int maxMem        = props->getTotalMemory() - 10 * 1048576;
    // const int maxReq  = maxStaticMem + maxKeySpace + maxValSpace + maxCountSpace + commSize * sizeof(int) * 4;
    numBuffers        = 1; // (accumMap ? 1 : std::min(static_cast<int>(chunks.size()), maxMem / maxReq));

    delete props;
    props = NULL;

    determineMaximumSpaceRequirements();

    keySize = valSize = 0;
    keySpace = maxKeySpace;
    valSpace = maxValSpace;

    allocateMapVariables();
    startBinnerThread();
    MPI_Barrier(MPI_COMM_WORLD);
    mapTimer.start();
    enqueueAllChunks();

    mapTimer.stop();
    mapPostTimer.start();
    binningTimer.start();
    partitionCheckSends(true);
    collectVariablesFromBinnerAndKill();
    binningTimer.stop();

    mapFreeTimer.start();
    freeMapVariables();
    mapFreeTimer.stop();
    mapPostTimer.stop();
  }
  void FixedSizeMapReduceJob::sort()
  {
    numUniqueKeys = -1;
    if (sorter->canExecuteOnGPU())  sorter->executeOnGPUAsync(keys, vals, keySize / sizeof(int), numUniqueKeys, &keyOffsets, &valOffsets, &numVals);
    else                            sorter->executeOnCPUAsync(keys, vals, valSize / sizeof(int), numUniqueKeys, &keyOffsets, &valOffsets, &numVals);
  }
  void FixedSizeMapReduceJob::reduce()
  {
    reduceTimer.start();

    emitConfigs.clear();
    keyCount.clear();
    configs.clear();
    cudacpp::DeviceProperties * props = cudacpp::DeviceProperties::get(getDeviceNumber());

    kernelStream = memcpyStream = &cudacpp::Stream::nullStream;

    getReduceRunParameters();
    // const int maxMem = props->getTotalMemory() - 10 * 1048576;
    // const int maxReq = maxInputKeySpace + maxInputValSpace + maxKeySpace + maxValSpace;
    // const int numBuffers  = std::min(static_cast<int>(chunks.size()), maxMem / maxReq);
    numBuffers = 1;
    delete props;
    props = NULL;

    allocateReduceVariables();

    configs.resize(numBuffers);
    for (unsigned int i = 0; i < configs.size(); ++i)
    {
      configs[i].keySpace   = reinterpret_cast<char * >(gpuKeys) + maxKeySpace * (i % numBuffers);
      configs[i].valueSpace = reinterpret_cast<char * >(gpuVals) + maxValSpace * (i % numBuffers);
    }

    enqueueReductions();

    reduceTimer.stop();

    cudacpp::Runtime::sync();

    freeReduceVariables();
  }
  void FixedSizeMapReduceJob::collectTimings()
  {
    /*
    const char * const descriptions[] =
    {
      "map",
      "bin",
      "sort",
      "reduce",
      "total",
      // "mapfree",
      // "mappost",
      // "fullmap",
      // "fullreduce",
      // "fulltime",
    };
    */
    double times[] =
    {
      mapTimer.getElapsedSeconds(),
      binningTimer.getElapsedSeconds(),
      sortTimer.getElapsedSeconds(),
      reduceTimer.getElapsedSeconds(),
      totalTimer.getElapsedSeconds(),
      // mapFreeTimer.getElapsedSeconds(),
      // mapPostTimer.getElapsedSeconds(),
      // fullMapTimer.getElapsedSeconds(),
      // fullReduceTimer.getElapsedSeconds(),
      // fullTimer.getElapsedSeconds(),
    };
    const int NUM_TIMES = sizeof(times) / sizeof(double);
    double * allTimes = new double[commSize * NUM_TIMES];
    MPI_Gather(times, NUM_TIMES, MPI_DOUBLE, allTimes, NUM_TIMES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (commRank != 0)
    {
      delete [] allTimes;
      return;
    }
    for (int i = 0; i < NUM_TIMES; ++i)
    {
      double min = times[i], max = times[i];
      double sum = 0.0f;
      for (int j = 0; j < commSize; ++j)
      {
        const double f = allTimes[j * NUM_TIMES + i];
        sum += f;
        min = std::min(min, f);
        max = std::max(max, f);
      }
      // printf("%-6s %10.3f %10.3f %10.3f\n", descriptions[i], min, max, sum / static_cast<double>(commSize));
      // printf(" %s=%5.3f", descriptions[i], sum / static_cast<double>(commSize));
      printf(" %5.3f", sum / static_cast<double>(commSize));
    }
    printf("\n");
    fflush(stdout);
    delete [] allTimes;
  }

  FixedSizeMapReduceJob::FixedSizeMapReduceJob(int & argc,
                                               char **& argv,
                                               const bool accumulateMapResults,
                                               const bool accumulateReduceResults,
                                               const bool syncOnPartitionSends)
    : MapReduceJob(argc, argv)
  {
    accumMap = accumulateMapResults;
    accumReduce = accumulateReduceResults;
    syncPartSends = syncOnPartitionSends;
  }
  FixedSizeMapReduceJob::~FixedSizeMapReduceJob()
  {
  }

  void FixedSizeMapReduceJob::addInput(Chunk * chunk)
  {
    chunks.push_back(chunk);
  }
  void FixedSizeMapReduceJob::execute()
  {
    mapTimer.start();
    binningTimer.start();
    sortTimer.start();
    reduceTimer.start();
    totalTimer.start();

    mapTimer.stop();
    binningTimer.stop();
    sortTimer.stop();
    reduceTimer.stop();
    totalTimer.stop();

    fullTimer.start();
    fullMapTimer.start();
    if (mapper != NULL)
    {
      if (binner          != NULL) binner->init();
      if (partitioner     != NULL) partitioner->init();
      if (partialReducer  != NULL) partialReducer->init();
      if (combiner        != NULL) combiner->init();

      mapper->init();
      MPI_Barrier(MPI_COMM_WORLD);
      totalTimer.start();
      map();
      mapper->finalize();

      if (binner          != NULL) binner->finalize();
      if (partitioner     != NULL) partitioner->finalize();
      if (partialReducer  != NULL) partialReducer->finalize();
      if (combiner        != NULL) combiner->init();
    }
    fullMapTimer.stop();

    sortTimer.start();
    if (sorter != NULL)
    {
      sorter->init();
      sort();
      sorter->finalize();
    }
    sortTimer.stop();

    fullReduceTimer.start();
    if (reducer != NULL && numUniqueKeys > 0)
    {
      reducer->init();
      reduce();
      totalTimer.stop();
      reducer->finalize();
    }
    fullReduceTimer.stop();
    fullTimer.stop();

    collectTimings();
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
