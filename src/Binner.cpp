#include <gpmr/Binner.h>

#include <mpi.h>

namespace gpmr
{
  Binner::Binner() : commSize(-1), commRank(-1)
  {
  }
  Binner::~Binner()
  {
  }
  void Binner::init()
  {
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  }
}
