#ifndef __OSCPP_THREAD_H__
#define __OSCPP_THREAD_H__

namespace oscpp
{
  class Runnable;

  class Thread
  {
    protected:
      void * handle;
      volatile bool running;
      Runnable * runner;

      static void * startThread(void * vself);
    public:
      Thread(Runnable * const pRunner);
      ~Thread();

      static void yield();
      void start();
      void run();
      void join();
      bool isRunning();
  };
}

#endif
