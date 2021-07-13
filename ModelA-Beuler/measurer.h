#ifndef MEASURER
#define MEASURER

#include "ModelA.h"
#include "make_unique.h"
#include <array>

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

  void measure(Vec *solution) {

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    computeSliceAverage(solution);
    if (rank == 0) {
      computeDerivedObs();
      measurer_out->save("phi");
      measurer_out->update();
    }
  }




    void computeEnergy(const double &dt)
    {
      DM da = model->domain;
      // Get a local vector with ghost cells
      Vec localUOld;
      DMGetLocalVector(da, &localUOld);
      Vec localUNew;
      DMGetLocalVector(da, &localUNew);

      // Fill in the ghost celss with mpicalls
      DMGlobalToLocalBegin(da, model->previoussolution, INSERT_VALUES, localUOld);
      DMGlobalToLocalEnd(da, model->previoussolution, INSERT_VALUES, localUOld);
      DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localUNew);
      DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localUNew);

      const ModelAData &data = model->data;

      G_node ***phiOld;
      DMDAVecGetArrayRead(da, localUOld, &phiOld);

      G_node ***phiNew;
      DMDAVecGetArrayRead(da, localUNew, &phiNew);


      const PetscReal H[4] = {data.H, 0., 0., 0.};


      PetscInt  xstart, ystart, zstart, xdimension, ydimension,
          zdimension;
      DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                     &zdimension);

      // Loop over central elements
      PetscScalar phimid = 0, grad2 = 0, nab2 = 0, hEn = 0;
      std::array<PetscScalar,4> localEnergy{0,0,0,0};


      for (PetscInt k = zstart; k < zstart + zdimension; k++) {
        for (PetscInt j = ystart; j < ystart + ydimension; j++) {
          for (PetscInt i = xstart; i < xstart + xdimension; i++) {
            for(int s = 0; s < ModelAData::Nphi; ++s){
              phimid = 0.5 * (phiNew[k][j][i].f[s] + phiOld[k][j][i].f[s]);

              grad2 += pow( 0.5 * (phiNew[k+1][j][i].f[s] + phiOld[k+1][j][i].f[s]) - phimid, 2);
              grad2 += pow( 0.5 * (phiNew[k][j+1][i].f[s] + phiOld[k][j+1][i].f[s]) - phimid, 2);
              grad2 += pow( 0.5 * (phiNew[k][j][i+1].f[s] + phiOld[k][j][i+1].f[s]) - phimid, 2);

              hEn += phimid * H[s];
            }
            for(PetscInt s=0;s < ModelAData::NV; s++ ){
              nab2 += pow(phiNew[k][j][i].V[s], 2);
            }

            for(PetscInt s=0;s < ModelAData::NA; s++ ){
              nab2 += pow(phiNew[k][j][i].A[s], 2);
            }

            localEnergy[0] +=  0.5 / data.chi * nab2;
            localEnergy[1] +=  0.5 * grad2;
            localEnergy[2] +=   hEn;
          }
        }
      }
      energy = std::array<PetscScalar,4>{0,0,0,0};
      MPI_Reduce(localEnergy.data(), energy.data(), energy.size() - 1, MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);

      for(PetscScalar& x : energy) x/= (data.NX * data.NY * data.NZ);
      energy.back() = energy[0] + energy[1] - energy[2];

      DMDAVecRestoreArrayRead(da, localUOld, &phiOld);
      DMRestoreLocalVector(da, &localUOld);
      DMDAVecRestoreArrayRead(da, localUNew, &phiNew);
      DMRestoreLocalVector(da, &localUNew);

    }

    std::array<PetscScalar,4> getEnergy() const{
      return energy;
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
    G_node ***fld;
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

    int actualInd = 0;

    // Store the local averages
    for (int k = izs; k < izs + nz; k++) {
      for (int j = iys; j < iys + ny; j++) {
        for (int i = ixs; i < ixs + nx; i++) {
          norm = 0.0;
          for (int l = 0; l < ModelAData::Nphi; l++) {
            sliceAveragesLocalX[l][i] += fld[k][j][i].f[l];
            sliceAveragesLocalY[l][j] += fld[k][j][i].f[l];
            sliceAveragesLocalZ[l][k] += fld[k][j][i].f[l];
            norm += pow(fld[k][j][i].f[l], 2);
          }
          norm = sqrt(norm);
          for (int l = ModelAData::Nphi; l < ModelAData::Nphi + ModelAData::NA; l++) {
            actualInd = l - ModelAData::Nphi;
            sliceAveragesLocalX[l][i] += fld[k][j][i].A[actualInd];
            sliceAveragesLocalY[l][j] += fld[k][j][i].A[actualInd];
            sliceAveragesLocalZ[l][k] += fld[k][j][i].A[actualInd];
          }
          for (int l = ModelAData::Nphi + ModelAData::NA; l < ModelAData::Nphi + ModelAData::NA + ModelAData::NV; l++) {
            actualInd = l - ModelAData::Nphi - ModelAData::NA;
            sliceAveragesLocalX[l][i] += fld[k][j][i].V[actualInd];
            sliceAveragesLocalY[l][j] += fld[k][j][i].V[actualInd];
            sliceAveragesLocalZ[l][k] += fld[k][j][i].V[actualInd];
          }
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
    for (int l = 0; l < ModelAData::Nphi; l++) {
      M2 += pow(OAverage[l], 2);
    }
    OAverage.at(NObs) = M2;
    OAverage.at(NObs + 1) = M2 * M2;
  }

  ModelA *model;
  PetscInt N;

  // Arrays of size Nobs contain X=(phi[1..Nphi], q[1...Nq], phi2)
  static const PetscInt NObs = ModelAData::Nphi + ModelAData::NA + ModelAData::NV + 1;

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

  std::array<PetscScalar, 4> energy;

  friend class measurer_output_hdf5;
  friend class measurer_output_txt;
  std::unique_ptr<measurer_output> measurer_out;
};

#endif
