#ifndef MEASURER
#define MEASURER

#include "ModelA.h"

class Measurer {
public:
  Measurer(ModelA *ptr) : model(ptr) {

    N = model->data.NX;
    if (model->data.NX != model->data.NY || model->data.NX != model->data.NZ) {
      PetscPrintf(
          PETSC_COMM_WORLD,
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
      throw(
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }

    // Create temporary petsc arrays for holding scalars
    VecCreateSeq(PETSC_COMM_SELF, NScalars, &vecSaverScalars);
    VecCreateSeq(PETSC_COMM_SELF, N, &vecSaverCors);

    // Set the parameters controlling the output
    verbosity = model->data.verboseMeasurements;
    outputfileName = model->data.outputfiletag + ".h5";

    // //These would be used for plotting. But this is not being used.
    //
    // currentTime = model->data.initialtime;
    // dt = model->data.deltat;

    // Create the viewer
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      std::string name(model->data.outputfiletag + ".h5");
      PetscViewerHDF5Open(PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE,
                          &viewer);

      PetscViewerSetFromOptions(viewer);
      PetscViewerHDF5SetTimestep(viewer, model->data.initialtime);
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

  void measure(Vec *solution, Vec *momenta) {

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

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
    }
  }

  /* // Used at one point for plotting the solution
  void openHDF5(std::string name, PetscReal time, int rank)
  {
    if(rank == 0) {
      PetscViewerHDF5Open(PETSC_COMM_SELF, name.c_str(), FILE_MODE_APPEND,
  &viewer); PetscViewerSetFromOptions(viewer);
      PetscViewerHDF5SetTimestep(viewer, time);
    }
  }

  void closeHDF5(int rank){
    if(rank == 0) PetscViewerDestroy(&viewer);
  }


  void savesolution(Vec *solution) {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    //openHDF5(outputfileName, currentTime, rank);;

    if(rank == 0){
        PetscObjectSetName((PetscObject) solution, "solution")
        VecView(solution,viewer)
        PetscViewerHDF5IncrementTimestep(viewer);
      //currentTime++;//=dt;
    }
  } */

private:
  void computeSliceAverage(Vec *solution) {
    // Get the local information and store in info

    DM &da = model->domain;
    Vec localU;
    DMGetLocalVector(da, &localU);
    // take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da, *solution, INSERT_VALUES, localU);
    DMGlobalToLocalEnd(da, *solution, INSERT_VALUES, localU);

    // From the vector define the pointer for the field phi
    o4_node ***fld;
    DMDAVecGetArrayRead(da, localU, &fld);

    // Set up the slize averages initialized to zero in c++11
    sliceAveragesX = std::vector<std::vector<PetscScalar>>(
        NObs, std::vector<PetscScalar>(N));
    sliceAveragesY = std::vector<std::vector<PetscScalar>>(
        NObs, std::vector<PetscScalar>(N));
    sliceAveragesZ = std::vector<std::vector<PetscScalar>>(
        NObs, std::vector<PetscScalar>(N));

    // Local arrays with same characteristic
    auto sliceAveragesLocalX = sliceAveragesX;
    auto sliceAveragesLocalY = sliceAveragesY;
    auto sliceAveragesLocalZ = sliceAveragesZ;

    // Get the ranges
    PetscInt ixs, iys, izs, nx, ny, nz;
    DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
    PetscReal norm;

    // Store the local averages
    for (int k = izs; k < izs + nz; k++) {
      for (int j = iys; j < iys + ny; j++) {
        for (int i = ixs; i < ixs + nx; i++) {
          norm = 0.0;
          for (int l = 0; l < ModelAData::Ndof; l++) {
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

    // Create the correlator array initialized to zero
    isotropicWallToWallCii = std::vector<std::vector<PetscScalar>>(
        NObs, std::vector<PetscScalar>(N));

    // Initializing the average to zero
    OAverage = std::vector<PetscScalar>(NScalars);

    // Compute the spatial correlation function of wall averages
    for (int l = 0; l < NObs; l++) {
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
    // Compute <X>
    for (int l = 0; l < NObs; l++) {
      OAverage[l] /= PetscReal(N);
    }

    // Append the magnetization M2 and (M**2)^2 to the scalar observable
    // vector this could be done offline
    PetscReal M2 = 0.;
    for (int l = 0; l < ModelAData::Ndof; l++) {
      M2 += pow(OAverage[l], 2);
    }
    OAverage.at(NObs) = M2;
    OAverage.at(NObs + 1) = M2 * M2;
  }

  void save(std::string fld) {
    std::string tmp;
    for (PetscInt i = 0; i < NObs; ++i) {
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

  // Save instance of a correlator like object into a hdf5 file
  void saveCorLike(std::vector<PetscScalar> &arr, std::string name) {

    // For indexing the Petsc objects where these will be saved
    std::vector<PetscInt> vecCorsIndex;
    for (PetscInt i = 0; i < N; ++i) {
      vecCorsIndex.emplace_back(i);
    }

    PetscObjectSetName((PetscObject)vecSaverCors, name.c_str());
    VecSetValues(vecSaverCors, N, vecCorsIndex.data(), arr.data(),
                 INSERT_VALUES);

    VecAssemblyBegin(vecSaverCors);
    VecAssemblyEnd(vecSaverCors);
    VecView(vecSaverCors, viewer);
  }

  // Save instance of scalars like object into a hdf5 file
  void saveScalarsLike(std::vector<PetscScalar> &arr, std::string name) {

    // for indixing the petsc vectors where these will be saved
    std::vector<PetscInt> vecScalarsIndex;
    for (PetscInt i = 0; i < NScalars; ++i) {
      vecScalarsIndex.emplace_back(i);
    }

    PetscObjectSetName((PetscObject)vecSaverScalars, name.c_str());

    VecSetValues(vecSaverScalars, NScalars, vecScalarsIndex.data(), arr.data(),
                 INSERT_VALUES);

    VecAssemblyBegin(vecSaverScalars);
    VecAssemblyEnd(vecSaverScalars);
    VecView(vecSaverScalars, viewer);
  }

  ModelA *model;
  PetscInt N;

  // Arrays of size Nobs contain X=(phi[0], phi[1], phi[2], phi[3], phi2)
  static const PetscInt NObs = ModelAData::Ndof + 1;

  std::vector<std::vector<PetscScalar>>
      sliceAveragesX; // First dimension is NObs, last dimension slice
                      // number.
  std::vector<std::vector<PetscScalar>>
      sliceAveragesY; // First dimension is NObs, last dimension slice
                      // number.
  std::vector<std::vector<PetscScalar>>
      sliceAveragesZ; // First dimension is Nobs, last dimension slice
                      // number.

  // An array of dimension Nobs calculating the isotropic wall wall auto
  // correlation functions of the observables, such as <X[2]*X[2]>
  std::vector<std::vector<PetscScalar>> isotropicWallToWallCii;

  // This contains the average of the five observables and two more.
  // Let M[l]  = 1/N^3  Sum phi[l].  The two additional observables are M^2 and
  // (M^2)^2
  static const PetscInt NScalars = NObs + 2;
  std::vector<PetscScalar> OAverage;

  // Controling the output
  PetscViewer viewer;
  bool verbosity;
  std::string outputfileName;

  // These variables are for plotting but aren't being used.
  //
  // PetscReal dt;
  // PetscInt currentTime;

  // These temporary local petsc vectors are to simplify the using petsc
  // routines for writing to HDF5. See the routines saveCorLike and
  // saveScalarsLike
  Vec vecSaverCors;
  Vec vecSaverScalars;
};

#endif
