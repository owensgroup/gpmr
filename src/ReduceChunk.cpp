#include <ReduceChunk.h>

namespace gpmr
{
  class ReduceChunk : public Chunk
  {
    public:
      ReduceChunk();
      virtual ~ReduceChunk();

      virtual bool updateExecutionQueuePosition(const int newPosition);
      virtual bool canExecuteOnCPU() const;
      virtual bool canExecuteOnGPU() const;
      virtual int getMemoryRequirementsOnGPU() const;
      virtual void stageAsync(void * const gpuStorage);
      virtual void finalizeAsync();
  };
}

#endif
