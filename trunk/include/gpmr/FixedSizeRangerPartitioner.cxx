#ifndef __GPMR_FIXEDSIZERANGEPARTITIONER_CXX__
#define __GPMR_FIXEDSIZERANGEPARTITIONER_CXX__

#include <gpmr/FixedSizeRangePartitioner.h>

#include <mpi.h>

namespace gpmr
{
  class EmitConfiguration;

  template <typename Key, typename Val> void gpmrFixedSizeRangePartitionerZeroCountsAndIndices(const int commSize, int * gpuBucketCounts, int * gpuBucketIndices, cudaStream_t * stream);
  template <typename Key, typename Val> void gpmrFixedSizeRangePartitionerCount               (const int commSize, const int numKeys, Key * gpuKeys, int * gpuBucketCounts, cudaStream_t * stream);
  template <typename Key, typename Val> void gpmrFixedSizeRangePartitionerPartitionToTemp     (const int commSize, const int numKeys, Key * gpuKeys, Value * gpuVals, Key * gpuTempKeySpace, Vakye * gpuTempValSpace, int * gpuKeyOffsets, int * gpuBucketIndices, cudaStream_t * stream);
  template <typename Key, typename Val> void gpmrFixedSizeRangePartitionerMoveFromTemp        (const int commSize, const int numKeys, Key * gpuKeys, Value * gpuVals, Key * gpuTempKeySpace, Vakye * gpuTempValSpace, cudaStream_t * stream);
  template <typename Key, typename Val> void gpmrFixedSizeRangePartitionerSetKeyAndValCounts  (const int commSize, int * gpuKeyCounts, int * gpuValCounts, cudaStream_t * stream);
  template <typename Key, typename Val> void gpmrFixedSizeRangePartitionerSetKeyAndValOffsets (const int commSize, const int * const gpuKeyCounts, int * const gpuKeyOffsets, int * const gpuValOffsets, cudaStream_t * stream);

  template <typename Key, typename Value>
  FixedSizeRangePartitioner<Key, Value>::FixedSizeRangePartitioner(const Key & pRangeBegin, const Key & pRangeEnd)
  {
    rangeBegin = pRangeBegin;
    rangeEnd   = pRangeEnd;
  }
  template <typename Key, typename Value>
  FixedSizeRangePartitioner<Key, Value>::~FixedSizeRangePartitioner()
  {
  }

  template <typename Key, typename Value>
  bool FixedSizeRangePartitioner<Key, Value>::canExecuteOnGPU() const
  {
    return true;
  }

  template <typename Key, typename Value>
  bool FixedSizeRangePartitioner<Key, Value>::canExecuteOnCPU() const
  {
    return false;
  }

  template <typename Key, typename Value>
  int  FixedSizeRangePartitioner<Key, Value>::getMemoryRequirementsOnGPU(gpmr::EmitConfiguration & emitConfig) const
  {
    return emitConfig.getKeySpace() + emitConfig.getValueSpace();
  }

  template <typename Key, typename Value>
  void FixedSizeRangePartitioner<Key, Value>::init()
  {
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  }

  template <typename Key, typename Value>
  void FixedSizeRangePartitioner<Key, Value>::finalize()
  {
  }

  template <typename Key, typename Value>
  void FixedSizeRangePartitioner<Key, Value>::executeOnGPUAsync(GPMRGPUConfig gpmrGPUConfig,
                                                                int * gpuKeyOffsets, int * gpuValOffsets,
                                                                int * gpuKeyCounts,  int * gpuValCounts,
                                                                gpmr::EmitConfiguration & emitConfig,
                                                                void * const gpuMemory,
                                                                cudacpp::Stream * kernelStream)
  {
    const int numKeys = gpmrGPUConfig.emitInfo.grid.numThreads * gpmrGPUConfig.emitInfo.grid.emitsPerThread;
    Key   * gpuTempKeySpace   = reinterpret_cast<Key * >(gpuMemory);
    Value * gpuTempValSpace   = reinterpret_cast<Value * >(gpuTempKeySpace + emitConfig.getIndexCount());
    int * gpuBucketCounts   = gpuKeyCounts;
    int * gpuBucketIndices  = gpuValCounts;
    Key   * gpuKeys           = reinterpret_cast<Key   * >(gpmrGPUConfig.keySpace);
    Value * gpuVals           = reinterpret_cast<Value * >(gpmrGPUConfig.valueSpace);

    gpmrFixedSizeRangePartitionerZeroCountsAndIndices(commSize, gpuBucketCounts, gpuBucketIndices, &kernelStream->getHandle());
    gpmrFixedSizeRangePartitionerCount               (commSize, numKeys, gpuKeys, gpuBucketCounts, &kernelStream->getHandle());
    gpmrFixedSizeRangePartitionerSetKeyAndValOffsets (commSize, gpuKeyCounts, gpuKeyOffsets, gpuValOffsets, &kernelStream->getHandle());
    gpmrFixedSizeRangePartitionerPartitionToTemp     (commSize, numKeys, gpuKeys, gpuVals, gpuTempKeySpace, gpuTempValSpace, gpuKeyOffsets, gpuBucketIndices, &kernelStream->getHandle());
    gpmrFixedSizeRangePartitionerMoveFromTemp        (commSize, numKeys, gpuKeys, gpuVals, gpuTempKeySpace, gpuTempValSpace, &kernelStream->getHandle());
    gpmrFixedSizeRangePartitionerSetKeyAndValCounts  (commSize, gpuKeyCounts, gpuValCounts, &kernelStream->getHandle());
    gpmrFixedSizeRangePartitionerSetKeyAndValOffsets (commSize, gpuKeyCounts, gpuKeyOffsets, gpuValOffsets, &kernelStream->getHandle());
  }

  // even though this isn't supported, we need to add it. otherwise the compiler
  // will give lots of errors.
  template <typename Key, typename Value>
  void FixedSizeRangePartitioner<Key, Value>::executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets)
  {
    const float range = static_cast<float>(rangeEnd - rangeBegin);
    const int numItems = gpmrCPUConfig.emitInfo.grid.numThreads * gpmrCPUConfig.emitInfo.grid.emitsPerThread;
    Key   * keys = reinterpret_cast<Key * >(gpmrCPUConfig.keySpace);
    Value * vals = reinterpret_cast<Value * >(gpmrCPUConfig.valueSpace);

    for (int i = 0; i < commSize; ++i) keyOffsets[i] = valOffsets[i] = 0;

    int * keyCounts = new int[commSize];
    int * valCounts = new int[commSize];

    memset(keyCounts, 0, sizeof(int) * commSize);
    memset(valCounts, 0, sizeof(int) * commSize);

    for (int i = 0; i < numItems; ++i)
    {
      const float        f     = static_cast<float>(keys[i] - rangeBegin) / range;
      const unsigned int index = static_cast<unsigned int>(f * commSize);
      ++keyCounts[index];
      ++valCounts[index];
    }

    keyOffsets[0] = valOffsets[0] = 0;
    for (int i = 1; i < commSize; ++i)
    {
      keyOffsets[i] = keyOffsets[i - 1] + keyCounts[i - 1];
      valOffsets[i] = valOffsets[i - 1] + valCounts[i - 1];
    }

    Key   * tempKeys = new Key  [numItems];
    Value * tempVals = new Value[numItems];

    memset(keyCounts, 0, sizeof(int) * commSize);
    for (int i = 0; i < numItems; ++i)
    {
      const float        f     = static_cast<float>(keys[i] - rangeBegin) / range;
      const unsigned int index = static_cast<unsigned int>(f * commSize);
      tempKeys[keyOffsets[index] + keyCounts[index]] = keys[i];
      tempVals[keyOffsets[index] + keyCounts[index]] = vals[i];
      keyCounts[index]++;
    }

    memcpy(keys, tempKeys, sizeof(Key)   * numItems);
    memcpy(vals, tempVals, sizeof(Value) * numItems);

    delete [] tempKeys;
    delete [] tempVals;
  }
}

#endif
