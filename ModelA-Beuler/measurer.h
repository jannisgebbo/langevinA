#ifndef MEASURER
#define MEASURER

#include "ModelA.h"
#include "make_unique.h"

class Measurer;

// Interface for output of measurements
class measurer_output {
public:
  virtual ~measurer_output() { ; }
  virtual void update() = 0;
  virtual void save(const std::string &what) = 0;
};

#ifndef MODELA_NO_HDF5
class measurer_output_hdf5 : public measurer_output {
public:
  measurer_output_hdf5(Measurer *in);
  ~measurer_output_hdf5();
  virtual void update();
  virtual void save(const std::string &what);

private:
  Measurer *measure;
  PetscViewer viewer;
  bool verbosity;

  // These temporary local petsc vectors are to simplify the using petsc
  // routines for writing to HDF5. See the routines saveCorLike and
  // saveScalarsLike
  Vec vecSaverCors;
  Vec vecSaverScalars;

  void saveScalarsLike(std::vector<PetscScalar> &arr, const std::string &name);
  void saveCorLike(std::vector<PetscScalar> &arr, const std::string &name);
};
#endif

// Minimal output of scalar array
class measurer_output_txt : public measurer_output {
public:
  measurer_output_txt(Measurer *in);
  ~measurer_output_txt();
  virtual void update() { ; }
  virtual void save(const std::string &what);

private:
  Measurer *measure;
  PetscViewer averages_asciiviewer;
};

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

    // Create the viewer
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
#ifndef MODELA_NO_HDF5
      measurer_out = make_unique<measurer_output_hdf5>(this);
#else
      measurer_out = make_unique<measurer_output_txt>(this);
#endif
    }
  }

  virtual ~Measurer() {}

  void finalize() {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      measurer_out.reset();
    }
  }

  void measure(Vec *solution, Vec *momenta) {

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    computeSliceAverage(solution);
    if (rank == 0) {
      computeDerivedObs();
      measurer_out->save("phi");
    }
    computeSliceAverage(momenta);
    if (rank == 0) {
      computeDerivedObs();
      measurer_out->save("phidot");
      measurer_out->update();
    }
  }

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

  friend class measurer_output_hdf5;
  friend class measurer_output_txt;
  std::unique_ptr<measurer_output> measurer_out;
};

#endif
