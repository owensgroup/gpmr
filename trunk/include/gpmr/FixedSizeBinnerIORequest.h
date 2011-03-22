#ifndef __GPMR_FIXEDSIZEBINNERIOREQUEST_H__
#define __GPMR_FIXEDSIZEBINNERIOREQUEST_H__

#include <oscpp/AsyncIORequest.h>
#include <oscpp/Condition.h>

#include <mpi.h>

namespace gpmr
{

  class FixedSizeBinnerIORequest : public oscpp::AsyncIORequest
  {
    protected:
      volatile bool * flag;
      volatile bool * waiting;
      int byteCount;
      oscpp::Condition cond;

      FixedSizeBinnerIORequest(const FixedSizeBinnerIORequest & rhs);
      FixedSizeBinnerIORequest & operator = (const FixedSizeBinnerIORequest & rhs);
    public:
      FixedSizeBinnerIORequest(volatile bool * const pFlag, volatile bool * const pWaiting, const int pByteCount);
      virtual ~FixedSizeBinnerIORequest();

      virtual bool query();
      virtual void sync();
      virtual bool hasError();
      virtual int bytesTransferedCount();

      inline oscpp::Condition & condition() { return cond; }
  };
}

#endif
