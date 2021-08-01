#ifndef NOISEGENERATOR
#define NOISEGENERATOR

#include <fstream>
#include <iostream>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>
#include <random>
#include <string>
#include <vector>

#define NOISEGENERATOR_STDCPP

#ifdef NOISEGENERATOR_STDCPP
#include "xoroshiro128plus.h"
#endif

#ifdef NOISEGENERATOR_GSL
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

class gsl_rng_urng {
public:
  typedef unsigned long int result_type;
  gsl_rng *grng;

  static constexpr unsigned long int min() { return 0; }
  static constexpr unsigned long int max() { return 0xFFFFFFFF; }
  gsl_rng_urng() { grng = gsl_rng_alloc(gsl_rng_mt19937); }
  ~gsl_rng_urng() { gsl_rng_free(grng); }
  void seed(unsigned long int sd) { gsl_rng_set(grng, sd); }

  unsigned long int operator()() { return gsl_rng_get(grng); }
  void write(const std::string &fname) {
    FILE *fp = fopen(fname.c_str(), "wb");
    if (!fp) {
      std::cout << "Unable to open file for writing in gsl URNG" << std::endl;
      std::terminate();
    } else {
      std::cout << "Writing  file in gsl URNG" << std::endl;
      gsl_rng_fwrite(fp, grng);
    }
    fclose(fp);
  }
  void read(const std::string &fname) {
    FILE *fp = fopen(fname.c_str(), "rb");
    if (!fp) {
      std::cout << "Unable to open file for reading in gsl URNG" << std::endl;
      std::terminate();
    } else {
      gsl_rng_fread(fp, grng);
    }
    fclose(fp);
  }
};
#endif

// typedef std::ranlux48 RNGType;
// typedef std::mt19937_64 RNGType;
typedef xoroshiro128plus RNGType;
// typedef gsl_rng_urng RNGType;

class NoiseGenerator {
public:
  NoiseGenerator(const int &baseSeed = 0)
      : uniformDistribution(-sqrt(3.), sqrt(3.)) {

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    int size = 0;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    std::seed_seq sequence = {baseSeed};
    std::vector<unsigned int> seeds(size, 0);
    sequence.generate(seeds.begin(), seeds.end());
    rng.seed(seeds[rank]);
  }

  ~NoiseGenerator() {}

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

  PetscReal normal() { return normalDistribution(rng); }

  PetscReal uniform() { return uniformDistribution(rng); }

  RNGType &generator() { return rng; }

  void write(const std::string &filename_stub) {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::string srank = std::to_string(rank);
    std::string fname = filename_stub + "_" + srank + ".rng";
#ifdef NOISEGENERATOR_GSL
    rng.write(fname);
#else
    std::ofstream out(fname);
    if (out) {
      out << rng;
    } else {
      std::cout << "Unable to open file when saving the random number "
                   "generator state!"
                << std::endl;
      std::terminate();
    }
#endif
  }

  void read(const std::string &filename_stub) {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::string srank = std::to_string(rank);
    std::string fname = filename_stub + "_" + srank + ".rng";
#ifdef NOISEGENERATOR_GSL
    rng.read(fname);
#else
    std::ifstream in(fname);
    if (in) {
      in >> rng;
    } else {
      std::cout << "Unable to open file when writing the random number "
                   "generator state!"
                << std::endl;
      std::terminate();
    }
#endif
  }

private:
  RNGType rng;
  std::normal_distribution<PetscReal> normalDistribution;
  std::uniform_real_distribution<PetscReal> uniformDistribution;
};

//! Global random number generator which is initialized when
//! ModelA is created.
extern std::unique_ptr<NoiseGenerator> ModelARndm;
#endif
