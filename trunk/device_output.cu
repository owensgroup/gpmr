#include <cuda.h>
#include <cstdlib>
#include <cstdio>

int main(int argc, char ** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    int deviceIndex = atoi(argv[i]);
    unsigned int memBytes;
    int major, minor;
    char name[1024], buf[1024];
    CUdevice dev;
    CUdevprop prop;
    cuInit(0);
    cuDeviceGet(&dev, deviceIndex);
    cuDeviceTotalMem(&memBytes, dev);
    cuDeviceComputeCapability(&major, &minor, dev);
    cuDeviceGetProperties(&prop, dev);
    cuDeviceGetName(name, 1023, dev);
    sprintf(buf,  "%d - %s:\n"
                  "  available mem:         %d bytes\n"
                  "  compute capability:    %d.%d\n"
                  "  max threads per block: %d\n"
                  "  max block dim:         { %d %d %d }\n"
                  "  max grid size:         { %d %d %d }\n"
                  "  shared mem per block:  %d bytes\n"
                  "  total constant mem:    %d bytes\n"
                  "  warp size:             %d threads\n"
                  "  memory pitch:          %d\n"
                  "  registers per block:   %d\n"
                  "  clock frequency:       %d kHz\n"
                  "  texture alignment:     %d\n",
                  deviceIndex, name, (int)memBytes, major, minor,
                  prop.maxThreadsPerBlock,
                  prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
                  prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
                  prop.sharedMemPerBlock,
                  prop.totalConstantMemory,
                  prop.SIMDWidth,
                  prop.memPitch,
                  prop.regsPerBlock,
                  prop.clockRate,
                  prop.textureAlign);
    printf("%s\n", buf); fflush(stdout);
  }

  return 0;
}
