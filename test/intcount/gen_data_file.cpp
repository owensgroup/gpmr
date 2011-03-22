#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "MersenneTwister.h"

int main(int argc, char ** argv)
{
  int numElems, seed, range;
  int * elems;
  if (argc != 5)
  {
    printf("Usage: %s output_file num_elements random_seed range\n", *argv);
    return 0;
  }
  FILE * fp = fopen(argv[1], "wb");
  MTRand mtrand(atoi(argv[3]));

  elems = new int[16 * 1024];
  numElems = atoi(argv[2]);
  range = atoi(argv[4]);
  int numLeft = numElems;
  printf("writing %s.\n", argv[1]); fflush(stdout);
  while (numLeft > 0)
  {
    printf("\r                                                        \r%d / %d", numElems - numLeft, numElems); fflush(stdout);
    int numToGen = std::min(16 * 1024, numLeft);
    numLeft -= numToGen;
    for (int i = 0; i < numToGen; ++i) elems[i] = mtrand.randInt(range);
    fwrite(elems, sizeof(int) * numToGen, 1, fp);
  }
  printf("\r                                                          \r%d / %d\ndone\n", numElems, numElems); fflush(stdout);

  fclose(fp);
}
