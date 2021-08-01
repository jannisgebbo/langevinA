#include "NoiseGenerator.h"
#include <petscsys.h>

int main(int argc, char **argv) {

  PetscErrorCode ierr;
  std::string help = "Random number tests";
  ierr = PetscInitialize(&argc, &argv, (char *)0, help.c_str());
  if (ierr) {
    return ierr;
  }
  int rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Read and Write
  int baseSeed = 2021;
  NoiseGenerator engine_w(baseSeed);
  for (size_t i = 0; i < 3; i++) {
    PetscScalar x = engine_w.uniform();
    if (i == 1) {
      engine_w.write("test_random");
    }
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d][%d] %f\n", i, rank, x);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
  }
  NoiseGenerator engine_r;
  engine_r.read("test_random");
  for (size_t i = 2; i < 3; i++) {
    PetscScalar x = engine_r.uniform();
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "x[%d][%d] %f\n", i, rank, x);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
  }
  // So the printing is not screwed up
  MPI_Barrier(PETSC_COMM_WORLD);

  NoiseGenerator engine(baseSeed);

  PetscLogEvent uniformGeneration;
  const size_t N = 1000000000;
  PetscLogEventRegister("Uniform", 0, &uniformGeneration);
  // Tests speed uniform
  PetscLogEventBegin(uniformGeneration, 0, 0, 0, 0);
  for (size_t i = 0; i < N; i++) {
    PetscScalar x = engine.uniform();
    if (i < 3) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] %f\n", rank, x);
      PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    }
  }
  PetscLogEventEnd(uniformGeneration, 0, 0, 0, 0);

  // So the printing is not screwed up
  MPI_Barrier(PETSC_COMM_WORLD);

  // Test speed normal
  PetscLogEvent normalGeneration;
  PetscLogEventRegister("Normal", 0, &normalGeneration);
  PetscLogEventBegin(normalGeneration, 0, 0, 0, 0);
  for (size_t i = 0; i < N; i++) {
    PetscScalar x = engine.normal();
    if (i < 3) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] %f\n", rank, x);
      PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    }
  }
  PetscLogEventEnd(normalGeneration, 0, 0, 0, 0);

  PetscFinalize();

  return 0;
}
