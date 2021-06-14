#ifndef NOISEGENERATOR
#define NOISEGENERATOR

#include <random>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>

class NoiseGenerator{
public :

NoiseGenerator(const int &baseSeed=0) 
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::seed_seq seq{baseSeed, rank}; 
  rng.seed(seq);
}

PetscErrorCode fillVec(Vec U)
{
   PetscScalar *array;
   int nloc = 0;
   VecGetLocalSize(U,&nloc);
   VecGetArray(U,&array);
   for (int i = 0; i < nloc; i++) {
     array[i] = normalDistribution(rng);
   }
   VecRestoreArray(U,&array);
   return 0;
}

PetscReal normal() { return normalDistribution(rng); }

private:

  typedef std::ranlux48 RNGType;
  RNGType rng;

  std::normal_distribution<double> normalDistribution;

};

//! Global random number generator which is initialized when
//! ModelA is created.
extern std::unique_ptr<NoiseGenerator> ModelARndm;
#endif
