#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include "MersenneTwister.h"

int main(int argc, char ** argv)
{
  FILE * fp;
  int numPoints;
  unsigned int seed;

  if (argc != 4)
  {
    printf("Usage: %s output_file num_points random_seed.\n", *argv);
    return 1;
  }
  fp = fopen(argv[1], "wb");
  if (!fp)
  {
    printf("Couldn't open %s for writing.\n", argv[1]);
    return 1;
  }
  if (sscanf(argv[2], "%d", &numPoints) != 1 || numPoints < 1)
  {
    printf("num_points must be an integer greater than zero, not '%s'.\n", argv[2]);
  }
  sscanf(argv[3], "%u", &seed);

  MTRand mtrand(seed);
  std::vector<float2> points;
  points.resize(numPoints);
  for (int i = 0; i < numPoints; ++i)
  {
    points[i].x = static_cast<float>(mtrand.rand());
    points[i].y = static_cast<float>(points[i].x + mtrand.rand(0.125) - 0.0625);
  }
  if (fwrite(&points[0], sizeof(float2) * numPoints, 1, fp) != 1) printf("Error writing.\n");
  fclose(fp);

  printf("Wrote %d points (%.3f MB) to %s.\n", numPoints, numPoints / 131072.0, argv[1]);

  return 0;
}
