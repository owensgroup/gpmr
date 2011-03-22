#include <gpmr/FixedSizeBinnerIORequest.h>

#include <oscpp/Thread.h>

namespace gpmr
{

  FixedSizeBinnerIORequest::FixedSizeBinnerIORequest(volatile bool * const pFlag, volatile bool * const pWaiting, const int pByteCount)
    : oscpp::AsyncIORequest(oscpp::AsyncIORequest::REQUEST_TYPE_WRITE)
  {
    flag = pFlag;
    waiting = pWaiting;
    byteCount = pByteCount;
  }
  FixedSizeBinnerIORequest::FixedSizeBinnerIORequest(const FixedSizeBinnerIORequest & rhs) : oscpp::AsyncIORequest(oscpp::AsyncIORequest::REQUEST_TYPE_WRITE)
  {
  }
  FixedSizeBinnerIORequest & FixedSizeBinnerIORequest::operator = (const FixedSizeBinnerIORequest & rhs)
  {
    return * this;
  }
  FixedSizeBinnerIORequest::~FixedSizeBinnerIORequest()
  {
    sync();
    delete flag;
  }

  bool FixedSizeBinnerIORequest::query()
  {
    return *flag;
  }
  void FixedSizeBinnerIORequest::sync()
  {
    cond.lockMutex();
    if (*flag)
    {
      cond.unlockMutex();
      return;
    }
    *waiting = true;
    cond.wait();
    *waiting = false;
    cond.unlockMutex();
  }
  bool FixedSizeBinnerIORequest::hasError()
  {
    return false;
  }
  int  FixedSizeBinnerIORequest::bytesTransferedCount()
  {
    return byteCount;
  }

}
