#ifndef MEASURER
#define MEASURER

#include "ModelA.h"

class Measurer {
public:
  Measurer(void *ptr) {
    ModelA *model = (ModelA *)ptr;
    da = model->domain;
    N = model->data.NX;
    Ndof = model->data.Ndof;
    verbosity = model->data.verboseMeasurements;

    if (model->data.NX != model->data.NY || model->data.NX != model->data.NZ) {
      PetscPrintf(
          PETSC_COMM_WORLD,
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
      throw(
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }
    int nObs = 5;
    VecCreateSeq(PETSC_COMM_SELF, nObs, &vecSaverScalars);
    VecCreateSeq(PETSC_COMM_SELF, N, &vecSaverCors);

    outputfileName = model->data.outputfiletag + ".h5";
    currentTime = model->data.initialtime;
    dt = model->data.deltat;
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      std::string name(model->data.outputfiletag + ".h5");
      PetscViewerHDF5Open(PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE,
                          &viewer);

      PetscViewerSetFromOptions(viewer);
      PetscViewerHDF5SetTimestep(viewer, model->data.initialtime);
    }

    for (PetscInt i = 0; i < N; ++i) {
      vecCorsIndex.emplace_back(i);
    }
    for (PetscInt i = 0; i < nObs; ++i) {
      vecScalarsIndex.emplace_back(i);
    }
  }

  virtual ~Measurer() {}

  void finalize() {
    VecDestroy(&vecSaverCors);
    VecDestroy(&vecSaverScalars);
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0)
      PetscViewerDestroy(&viewer);
  }

  /*void openHDF5(std::string name, PetscReal time, int rank)
  {
    if(rank == 0) {
      PetscViewerHDF5Open(PETSC_COMM_SELF, name.c_str(), FILE_MODE_APPEND,
  &viewer); PetscViewerSetFromOptions(viewer);
      PetscViewerHDF5SetTimestep(viewer, time);
    }
  }

  void closeHDF5(int rank){
    if(rank == 0) PetscViewerDestroy(&viewer);
  }*/

  void measure(Vec *solution, Vec *momenta) {

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // openHDF5(outputfileName, currentTime, rank);;

    computeSliceAverage(solution);
    if (rank == 0) {
      computeDerivedObs();
      save("phi");
    }
    computeSliceAverage(momenta);
    if (rank == 0) {
      computeDerivedObs();
      save("phidot");
      PetscViewerHDF5IncrementTimestep(viewer);
      // currentTime++;//=dt;
    }

    // closeHDF5(rank);
  }

  void savesolution(Vec *solution) {
    /*int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    //openHDF5(outputfileName, currentTime, rank);;

    if(rank == 0){
        PetscObjectSetName((PetscObject) solution, "solution")
        VecView(solution,viewer)
        PetscViewerHDF5IncrementTimestep(viewer);
      //currentTime++;//=dt;
    }*/
  }

private:
  void computeSliceAverage(Vec *solution) {
    // Get the local information and store in info

    Vec localU;
    DMGetLocalVector(da, &localU);
    // take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da, *solution, INSERT_VALUES, localU);
    DMGlobalToLocalEnd(da, *solution, INSERT_VALUES, localU);

    // From the vector define the pointer for the field phi
    o4_node ***fld;
    DMDAVecGetArrayRead(da, localU, &fld);

    // OX0[i] will contain the average of fld0 at wall x=i.
    // And analagous vectors for the wall averages in the y and z directions

    sliceAveragesX = std::vector<std::vector<PetscScalar>>(
        Ndof + 1, std::vector<PetscScalar>(N));
    sliceAveragesY = std::vector<std::vector<PetscScalar>>(
        Ndof + 1, std::vector<PetscScalar>(N));
    sliceAveragesZ = std::vector<std::vector<PetscScalar>>(
        Ndof + 1, std::vector<PetscScalar>(N));
    auto sliceAveragesLocalX = sliceAveragesX;
    auto sliceAveragesLocalY = sliceAveragesY;
    auto sliceAveragesLocalZ = sliceAveragesZ;

    // Get the ranges
    PetscInt ixs, iys, izs, nx, ny, nz;
    DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
    PetscReal norm;

    for (int k = izs; k < izs + nz; k++) {
      for (int j = iys; j < iys + ny; j++) {
        for (int i = ixs; i < ixs + nx; i++) {
          norm = 0.0;
          for (int l = 0; l < Ndof; l++) {
            sliceAveragesLocalX[l][i] += fld[k][j][i].f[l];
            sliceAveragesLocalY[l][j] += fld[k][j][i].f[l];
            sliceAveragesLocalZ[l][k] += fld[k][j][i].f[l];
            norm += pow(fld[k][j][i].f[l], 2);
          }
          norm = sqrt(norm);
          sliceAveragesLocalX.back()[i] += norm;
          sliceAveragesLocalY.back()[j] += norm;
          sliceAveragesLocalZ.back()[k] += norm;
        }
      }
    }

    for (int l = 0; l < sliceAveragesLocalX.size(); ++l) {
      for (int i = 0; i < N; ++i) {
        sliceAveragesLocalX[l][i] /= PetscReal(N * N);
        sliceAveragesLocalY[l][i] /= PetscReal(N * N);
        sliceAveragesLocalZ[l][i] /= PetscReal(N * N);
      }
    }
    // Retstore the array
    DMDAVecRestoreArrayRead(da, localU, &fld);
    DMRestoreLocalVector(da, &localU);

    // Bring all the data x data into one
    for (int l = 0; l < sliceAveragesLocalX.size(); l++) {
      MPI_Reduce(sliceAveragesLocalX[l].data(), sliceAveragesX[l].data(), N,
                 MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);
      MPI_Reduce(sliceAveragesLocalY[l].data(), sliceAveragesY[l].data(), N,
                 MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);
      MPI_Reduce(sliceAveragesLocalZ[l].data(), sliceAveragesZ[l].data(), N,
                 MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);
    }
  }

  void computeDerivedObs() {
    isotropicWallToWallCii = std::vector<std::vector<PetscScalar>>(
        Ndof + 1, std::vector<PetscScalar>(N));
    OAverage = std::vector<PetscScalar>(Ndof + 1);
    for (int l = 0; l < Ndof + 1; l++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          int jj = (i + j) % N; // Circulate the index, if i=N-1 j=1, jj=0

          isotropicWallToWallCii[l][j] +=
              (sliceAveragesX[l][i] * sliceAveragesX[l][jj] +
               sliceAveragesY[l][i] * sliceAveragesY[l][jj] +
               sliceAveragesZ[l][i] * sliceAveragesZ[l][jj]) /
              3.;
        }
        OAverage[l] += sliceAveragesX[l][i];
      }
    }
    for (int l = 0; l < Ndof + 1; l++) {
      OAverage[l] /= PetscReal(N);
    }
  }

  void save(std::string fld) {
    std::string tmp;
    for (PetscInt i = 0; i < Ndof + 1; ++i) {
      tmp = fld + "_" + std::to_string(i);
      saveCorLike(sliceAveragesX[i], "wallX_" + tmp);

      if (verbosity)
        saveCorLike(sliceAveragesY[i], "wallY_" + tmp);
      if (verbosity)
        saveCorLike(sliceAveragesZ[i], "wallZ_" + tmp);

      if (verbosity)
        saveCorLike(isotropicWallToWallCii[i], "Cii_" + tmp);
    }
    saveScalarsLike(OAverage, fld);
  }

  void saveCorLike(std::vector<PetscScalar> &arr, std::string name) {
    PetscObjectSetName((PetscObject)vecSaverCors, name.c_str());

    VecSetValues(vecSaverCors, N, vecCorsIndex.data(), arr.data(),
                 INSERT_VALUES);

    VecAssemblyBegin(vecSaverCors);
    VecAssemblyEnd(vecSaverCors);
    VecView(vecSaverCors, viewer);
  }

  void saveScalarsLike(std::vector<PetscScalar> &arr, std::string name) {
    PetscObjectSetName((PetscObject)vecSaverScalars, name.c_str());

    VecSetValues(vecSaverScalars, Ndof + 1, vecScalarsIndex.data(), arr.data(),
                 INSERT_VALUES);

    VecAssemblyBegin(vecSaverScalars);
    VecAssemblyEnd(vecSaverScalars);
    VecView(vecSaverScalars, viewer);
  }

  DM da;
  PetscInt N;
  PetscInt Ndof;
  Vec vecSaverCors;
  std::vector<PetscInt> vecCorsIndex;
  Vec vecSaverScalars;
  std::vector<PetscInt> vecScalarsIndex;

  bool verbosity;

  std::vector<std::vector<PetscScalar>>
      sliceAveragesX; // First dimension O4 component, last dimension slice
                      // number.
  std::vector<std::vector<PetscScalar>>
      sliceAveragesY; // First dimension O4 component, last dimension slice
                      // number.
  std::vector<std::vector<PetscScalar>>
      sliceAveragesZ; // First dimension O4 component, last dimension slice
                      // number.

  std::vector<std::vector<PetscScalar>> isotropicWallToWallCii;

  std::vector<PetscScalar> OAverage;
  std::vector<PetscScalar> OAverageLocal;

  // Viewer
  PetscViewer viewer;

  PetscReal dt;
  std::string outputfileName;
  PetscInt currentTime;
};

#endif
