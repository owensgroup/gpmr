#ifndef __GPMR_FIXEDSIZEBINNER_H__
#define __GPMR_FIXEDSIZEBINNER_H__

#include <gpmr/Binner.h>

#include <oscpp/Condition.h>
#include <oscpp/Mutex.h>

#include <mpi.h>

#include <list>
#include <vector>

namespace gpmr
{

  typedef struct _FixedSizeBinnerIOData
  {
    int rank;
    void * keys, * vals;
    int keySize, valSize;
    int * counts;
    volatile bool * flag;
    volatile bool * waiting;
    oscpp::Condition * cond;
    bool done[3];
    MPI_Request reqs[3];
    MPI_Status stat[3];
  } FixedSizeBinnerIOData;

  class FixedSizeBinner : public Binner
  {
    protected:
      oscpp::Mutex addDataLock;

      std::list<FixedSizeBinnerIOData * > needsToBeSent;
      std::list<FixedSizeBinnerIOData * > pendingIO;
      bool innerLoopDone;
      bool copySendData;
      int zeroCount[2];
      std::vector<MPI_Request> zeroReqs;

      int singleKeySize, singleValSize;
      char * finalKeys, * finalVals;
      int finalKeySize, finalValSize;
      int finalKeySpace, finalValSpace;

      bool pollUnsent();
      void pollPending();
      void pollSends();
      void poll(int & finishedWorkers,
                bool * const workerDone,
                bool * const recvingCount,
                int * const counts,
                int ** keyRecv,
                int ** valRecv,
                MPI_Request * recvReqs);
      void privateAdd(const void * const keys, const void * const vals, const int keySize, const int valSize);
      void grow(const int size, const int finalSize, int & finalSpace, char *& finals);

    public:
      FixedSizeBinner(const int pSingleKeySize, const int pSingleValSize, const bool pCopySendData = false);
      virtual ~FixedSizeBinner();

      virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             const int keySize,
                                             const int valSize);
      virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             int * const keySizes,
                                             int * const valSizes,
                                             const int numKeys,
                                             const int numVals);
      virtual void init();
      virtual void finalize();
      virtual void run();
      virtual oscpp::AsyncIORequest * finish();
      virtual void getFinalDataSize(int & keySize, int & valSize) const;
      virtual void getFinalDataSize(int & keySize, int & valSize, int & numKeys, int & numVals) const;
      virtual void getFinalData(void * keyStorage, void * valStorage) const;
      virtual void getFinalData(void * keyStorage, void * valStorage, int * keySizes, int * valSizes) const;
  };
}

#endif
