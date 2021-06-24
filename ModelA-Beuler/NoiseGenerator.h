#ifndef NOISEGENERATOR
#define NOISEGENERATOR

#include <memory>
#include <petscsys.h>
#include <petscvec.h>

#define NOISEGENERATOR_STDCPP

#ifdef NOISEGENERATOR_STDCPP
#include <random>
#include "xoroshiro128plus.h"
#endif
#ifdef NOISEGENERATOR_GSL
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#endif

class NoiseGenerator {
public:
  NoiseGenerator(const int &baseSeed = 0) {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

#ifdef NOISEGENERATOR_GSL
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, baseSeed + rank);
#endif
#ifdef NOISEGENERATOR_STDCPP
    //std::seed_seq seq{baseSeed, rank};
    //rng.seed(seq);
    rng.seed(baseSeed+rank);
#endif
  }

  ~NoiseGenerator() {
#ifdef NOISEGENERATOR_GSL
    gsl_rng_free(rng);
#endif
#ifdef NOISEGENERATOR_STDCPP
#endif
  }

  PetscErrorCode fillVec(Vec U) {
    PetscScalar *array;
    int nloc = 0;
    VecGetLocalSize(U, &nloc);
    VecGetArray(U, &array);

    for (int i = 0; i < nloc; i++) {
      array[i] = normal();
    }

    VecRestoreArray(U, &array);
    return 0;
  }

  PetscReal normal() {
#ifdef NOISEGENERATOR_GSL
    return gsl_ran_gaussian(rng, 1.);
#endif
#ifdef NOISEGENERATOR_STDCPP
    return normalDistribution(rng);
#endif
  }

  PetscReal uniform() {
#ifdef NOISEGENERATOR_GSL
    return gsl_rng_uniform(rng);
#endif
#ifdef NOISEGENERATOR_STDCPP
    return uniformDistribution(rng);
#endif
  }

private:
#ifdef NOISEGENERATOR_GSL
  gsl_rng *rng;
#endif
#ifdef NOISEGENERATOR_STDCPP
  //typedef std::ranlux48 RNGType;
  //typedef xoroshiro128plus RNGType;
  typedef std::mt19937_64 RNGType;
  RNGType rng;
  std::normal_distribution<PetscReal> normalDistribution;
  std::uniform_real_distribution<PetscReal> uniformDistribution;
#endif
};

//! Global random number generator which is initialized when
//! ModelA is created.
extern std::unique_ptr<NoiseGenerator> ModelARndm;
#endif

// Prodcues gaussian random numbers with mean 0 and variance 1
// void normal_pair(PetscReal &z1, PetscReal &z2) {
//  constexpr PetscReal two_pi = 2.0 * M_PI;
//
//  //create two random numbers, make sure u1 is greater than epsilon
//  PetscReal u1, u2;
//  u1 = gsl_rng_uniform_pos(rng) ;
//  u2 = gsl_rng_uniform(rng) ;

//  //compute z1 and z2
//  PetscReal mag = sqrt(-2.0 * log(u1));
//  z1  = mag * cos(two_pi * u2) ;
//  z2  = mag * sin(two_pi * u2) ;
//}
