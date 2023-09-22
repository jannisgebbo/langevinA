#include "measurer.h"
#include "nvector.h"
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifndef MODELA_NO_HDF5
/////////////////////////////////////////////////////////////////////////

measurer_output_fasthdf5::measurer_output_fasthdf5(Measurer *in) : measure(in) {

  const auto &ahandler = measure->model->data.ahandler;

  std::string name;

  if (ahandler.eventmode) {
    // Write an HDF5 file for each event call foo_000001.h5
    std::stringstream namestream;
    namestream << ahandler.outputfiletag << "_" << std::setw(4)
               << std::setfill('0') << ahandler.current_event << ".h5";
    name = namestream.str();
  } else {
    // This is box mode one hdf5 file for the run
    name = ahandler.outputfiletag + ".h5";
  }

  if (ahandler.eventmode) {
    // Each event is stored in a separate file
    file_id = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    // Check if we are supposed to, and are able to, restart the measurements.
    // Reopen up the hdf5 file and continue writing to tape
    PetscBool restartflg = PETSC_FALSE;
    if (ahandler.restart && !ahandler.eventmode) {
      PetscTestFile(name.c_str(), '\0', &restartflg);
    }

    if (restartflg and !ahandler.eventmode) {
      // File exists and we are in restart mode,  so open it for readwrite
      file_id = H5Fopen(name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    } else {
      // File either doesn't exist or we are not in restart mode, create a new
      // file
      file_id =
          H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
  }

  std::array<size_t, 1> NN1{Measurer::NScalars};
  scalars = std::make_unique<ntuple<1>>(NN1, "phi", file_id);
  NN1 = {2};
  timeout = std::make_unique<ntuple<1>>(NN1, "timeout", file_id);

  std::array<size_t, 2> NN2{Measurer::NObs, static_cast<size_t>(measure->N)};
  wallx = std::make_unique<ntuple<2>>(NN2, "wallx", file_id);
  wally = std::make_unique<ntuple<2>>(NN2, "wally", file_id);
  wallz = std::make_unique<ntuple<2>>(NN2, "wallz", file_id);

  std::array<size_t, 3> NN3 = {Measurer::NObs,
                               static_cast<size_t>(measure->N) / 2 + 1, 2};
  wallx_k = std::make_unique<ntuple<3>>(NN3, "wallx_k", file_id);
  wally_k = std::make_unique<ntuple<3>>(NN3, "wally_k", file_id);
  wallz_k = std::make_unique<ntuple<3>>(NN3, "wallz_k", file_id);

  NN3 = {Measurer::NObsRotated, static_cast<size_t>(measure->N) / 2 + 1, 2};
  wallx_k_rotated =
      std::make_unique<ntuple<3>>(NN3, "wallx_k_rotated", file_id);
  wally_k_rotated =
      std::make_unique<ntuple<3>>(NN3, "wally_k_rotated", file_id);
  wallz_k_rotated =
      std::make_unique<ntuple<3>>(NN3, "wallz_k_rotated", file_id);

  NN2 = {Measurer::NObsPhase, static_cast<size_t>(measure->N)};
  wallx_phase = std::make_unique<ntuple<2>>(NN2, "wallx_phase", file_id);
  wally_phase = std::make_unique<ntuple<2>>(NN2, "wally_phase", file_id);
  wallz_phase = std::make_unique<ntuple<2>>(NN2, "wallz_phase", file_id);

  NN3 = {Measurer::NObsPhase, static_cast<size_t>(measure->N) / 2 + 1, 2};
  wallx_phase_k = std::make_unique<ntuple<3>>(NN3, "wallx_phase_k", file_id);
  wally_phase_k = std::make_unique<ntuple<3>>(NN3, "wally_phase_k", file_id);
  wallz_phase_k = std::make_unique<ntuple<3>>(NN3, "wallz_phase_k", file_id);
}

measurer_output_fasthdf5::~measurer_output_fasthdf5() { H5Fclose(file_id); }

void measurer_output_fasthdf5::save(const std::string &what) {

  timeout->row[0] = measure->model->data.atime.t();
  timeout->row[1] = measure->model->data.mass();
  timeout->fill();

  scalars->row = measure->OAverage;
  scalars->fill();

  wallx->row = measure->wallX.v;
  wally->row = measure->wallY.v;
  wallz->row = measure->wallZ.v;

  // wallx->fill();
  // wally->fill();
  // wallz->fill();

  // hdf5 doesn't have complex data types uses arrays with extra dimension
  // [0],[1] sequentially in memory
  std::memcpy(wallx_k->row.data(), measure->wallX_k.v.data(),
              wallx_k->row.size() * sizeof(double));
  std::memcpy(wally_k->row.data(), measure->wallY_k.v.data(),
              wally_k->row.size() * sizeof(double));
  std::memcpy(wallz_k->row.data(), measure->wallZ_k.v.data(),
              wallz_k->row.size() * sizeof(double));

  wallx_k->fill();
  wally_k->fill();
  wallz_k->fill();

  std::memcpy(wallx_k_rotated->row.data(), measure->wallX_k_rotated.v.data(),
              wallx_k_rotated->row.size() * sizeof(double));
  std::memcpy(wally_k_rotated->row.data(), measure->wallY_k_rotated.v.data(),
              wally_k_rotated->row.size() * sizeof(double));
  std::memcpy(wallz_k_rotated->row.data(), measure->wallZ_k_rotated.v.data(),
              wallz_k_rotated->row.size() * sizeof(double));

  wallx_k_rotated->fill();
  wally_k_rotated->fill();
  wallz_k_rotated->fill();

  wallx_phase->row = measure->wallXPhase.v;
  wally_phase->row = measure->wallYPhase.v;
  wallz_phase->row = measure->wallZPhase.v;

  // wallx_phase->fill();
  // wally_phase->fill();
  // wallz_phase->fill();

  std::memcpy(wallx_phase_k->row.data(), measure->wallXPhase_k.v.data(),
              wallx_phase_k->row.size() * sizeof(double));
  std::memcpy(wally_phase_k->row.data(), measure->wallYPhase_k.v.data(),
              wally_phase_k->row.size() * sizeof(double));
  std::memcpy(wallz_phase_k->row.data(), measure->wallZPhase_k.v.data(),
              wallz_phase_k->row.size() * sizeof(double));

  wallx_phase_k->fill();
  wally_phase_k->fill();
  wallz_phase_k->fill();
}
#endif

////////////////////////////////////////////////////////////////////////

void Measurer::computeSliceAverage(Vec *solution) {
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
  std::fill(wallX.v.begin(), wallX.v.end(), 0.);
  std::fill(wallY.v.begin(), wallY.v.end(), 0.);
  std::fill(wallZ.v.begin(), wallZ.v.end(), 0.);

  // Local arrays with same characteristic dimensions
  nvector<PetscScalar, 2> wallLocalX(NObs, N);
  nvector<PetscScalar, 2> wallLocalY(NObs, N);
  nvector<PetscScalar, 2> wallLocalZ(NObs, N);

  std::fill(wallLocalX.v.begin(), wallLocalX.v.end(), 0.);
  std::fill(wallLocalY.v.begin(), wallLocalY.v.end(), 0.);
  std::fill(wallLocalZ.v.begin(), wallLocalZ.v.end(), 0.);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
  PetscReal phi2;

  int actualInd = 0;

  // Store the local averages
  for (int k = izs; k < izs + nz; k++) {
    for (int j = iys; j < iys + ny; j++) {
      for (int i = ixs; i < ixs + nx; i++) {
        phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          wallLocalX(l, i) += fld[k][j][i].f[l];
          wallLocalY(l, j) += fld[k][j][i].f[l];
          wallLocalZ(l, k) += fld[k][j][i].f[l];
          phi2 += pow(fld[k][j][i].f[l], 2);
        }
        for (int l = ModelAData::Nphi; l < ModelAData::Nphi + ModelAData::NA;
             l++) {
          actualInd = l - ModelAData::Nphi;
          wallLocalX(l, i) += fld[k][j][i].A[actualInd];
          wallLocalY(l, j) += fld[k][j][i].A[actualInd];
          wallLocalZ(l, k) += fld[k][j][i].A[actualInd];
        }
        for (int l = ModelAData::Nphi + ModelAData::NA;
             l < ModelAData::Nphi + ModelAData::NA + ModelAData::NV; l++) {
          actualInd = l - ModelAData::Nphi - ModelAData::NA;
          wallLocalX(l, i) += fld[k][j][i].V[actualInd];
          wallLocalY(l, j) += fld[k][j][i].V[actualInd];
          wallLocalZ(l, k) += fld[k][j][i].V[actualInd];
        }
        int l = ModelAData::Nphi + ModelAData::NA + ModelAData::NV;
        wallLocalX(l, i) += phi2;
        wallLocalY(l, j) += phi2;
        wallLocalZ(l, k) += phi2;
      }
    }
  }

  for (int l = 0; l < NObs; ++l) {
    for (int i = 0; i < N; ++i) {
      wallLocalX(l, i) /= PetscReal(N * N);
      wallLocalY(l, i) /= PetscReal(N * N);
      wallLocalZ(l, i) /= PetscReal(N * N);
    }
  }

  // Retstore the array
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);

  for (int l = 0; l < NObs; l++) {
    MPI_Reduce(&wallLocalX(l, 0), &wallX(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    MPI_Reduce(&wallLocalY(l, 0), &wallY(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    MPI_Reduce(&wallLocalZ(l, 0), &wallZ(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
  }
}

// Helper function for slice averages works for double and complex types
template <typename T>
inline std::array<T, 4> contract_rho(const std::array<T, 4> &n,
                                     const std::array<T, 6> &rho) {
  return {-(n[1] * rho[0]) - n[2] * rho[1] - n[3] * rho[2],
          n[0] * rho[0] + n[3] * rho[4] - n[2] * rho[5],
          n[0] * rho[1] - n[3] * rho[3] + n[1] * rho[5],
          n[0] * rho[2] + n[2] * rho[3] - n[1] * rho[4]};
}

// Helper function for slice averages works for double and complex types
template <typename T>
inline std::array<T, 6> dualize_rho(const std::array<T, 6> &rho) {
  return std::array<T, 6>{rho[3], rho[4], rho[5], rho[0], rho[1], rho[2]};
}

void Measurer::computeSliceAveragePhase(Vec *solution) {
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
  std::fill(wallXPhase.v.begin(), wallXPhase.v.end(), 0.);
  std::fill(wallYPhase.v.begin(), wallYPhase.v.end(), 0.);
  std::fill(wallZPhase.v.begin(), wallZPhase.v.end(), 0.);

  // Local arrays with same dimensions initialized to zero
  nvector<double, 2> wallXPhaseLocal(NObsPhase, N);
  nvector<double, 2> wallYPhaseLocal(NObsPhase, N);
  nvector<double, 2> wallZPhaseLocal(NObsPhase, N);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
  PetscReal phi2;

  double sigma;
  std::array<double, 4> n{};
  std::array<double, 4> phi{};
  std::array<double, 6> rho{};
  std::array<double, 4> V{};
  std::array<double, 4> A{};
  // Store the local averages
  for (int k = izs; k < izs + nz; k++) {
    for (int j = iys; j < iys + ny; j++) {
      for (int i = ixs; i < ixs + nx; i++) {

        double phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          phi[l] = fld[k][j][i].f[l];
          phi2 += pow(phi[l], 2);
        }
        sigma = sqrt(phi2);

        if (phi2 > 100. * std::numeric_limits<double>::min()) {
          for (int l = 0; l < ModelAData::Nphi; l++) {
            n[l] = phi[l] / sigma;
          }
        } else {
          n = {1, 0, 0, 0};
        }

        for (int l = 0; l < ModelAData::NA; l++) {
          rho[l] = fld[k][j][i].A[l];
          rho[l + ModelAData::NA] = fld[k][j][i].V[l];
        }
        A = contract_rho(n, rho);
        V = contract_rho(n, dualize_rho(rho));

        wallXPhaseLocal(0, i) += sigma;
        wallYPhaseLocal(0, j) += sigma;
        wallZPhaseLocal(0, k) += sigma;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          wallXPhaseLocal(1 + l, i) += n[l];
          wallYPhaseLocal(1 + l, j) += n[l];
          wallZPhaseLocal(1 + l, k) += n[l];

          wallXPhaseLocal(5 + l, i) += A[l];
          wallYPhaseLocal(5 + l, j) += A[l];
          wallZPhaseLocal(5 + l, k) += A[l];

          wallXPhaseLocal(9 + l, i) += V[l];
          wallYPhaseLocal(9 + l, j) += V[l];
          wallZPhaseLocal(9 + l, k) += V[l];
        }
        wallXPhaseLocal(Measurer::NObsPhase - 1, i) += sigma * sigma;
        wallYPhaseLocal(Measurer::NObsPhase - 1, j) += sigma * sigma;
        wallZPhaseLocal(Measurer::NObsPhase - 1, k) += sigma * sigma;
      }
    }
  }
  for (int l = 0; l < NObsPhase; ++l) {
    for (int i = 0; i < N; ++i) {
      wallXPhaseLocal(l, i) /= PetscReal(N * N);
      wallYPhaseLocal(l, i) /= PetscReal(N * N);
      wallZPhaseLocal(l, i) /= PetscReal(N * N);
    }
  }

  // Retstore the array
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);

  // Bring all the data x data into one
  for (int l = 0; l < NObsPhase; l++) {
    MPI_Reduce(&wallXPhaseLocal(l, 0), &wallXPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&wallYPhaseLocal(l, 0), &wallYPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&wallZPhaseLocal(l, 0), &wallZPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
  }
}

// Given the fourier transform of phi_a(t,k) and other fields
// stored in the wallk strucutre, compute the rotated fields
// The zero mode has phi(t,0) defines a unit four vector n_a
//
// wallk_rotated contains  sigma = n_a phi_a and pi_b = rho_{ab} n_b
// as well as the axial vector density  A_a = rho_{ab} n_b and
// corresponding vector density V_a. Finally phi2 is also
// stored in this structure
void rotate_to_vev(nvector<std::complex<double>, 2> &wallk,
                   nvector<std::complex<double>, 2> &wallk_rotated) {
  std::array<std::complex<double>, 4> n{};
  std::array<std::complex<double>, 4> phi{};
  std::array<std::complex<double>, 4> phir{};
  std::array<std::complex<double>, 6> rho{};
  std::array<std::complex<double>, 4> V{};
  std::array<std::complex<double>, 4> A{};

  // Extract the vev direction
  double norm = 0.;
  for (int a = 0; a < 4; a++) {
    n[a] = wallk(a, 0); // Extract the zero mode  of phi_a
    norm += std::norm(n[a]);
  }

  norm = sqrt(norm);
  if (norm < std::numeric_limits<double>::min()) {
    n = std::array<std::complex<double>, 4>{1, 0, 0, 0};
  } else {
    for (size_t a = 0; a < 4; a++) {
      n[a] /= norm;
    }
  }

  // Loop over momenta and do the rotation to vev
  for (size_t k = 0; k < wallk.N[1]; k++) {
    for (size_t a = 0; a < 4; a++) {
      phi[a] = wallk(a, k);
    }

    for (size_t ab = 0; ab < 6; ab++) {
      rho[ab] = wallk(4 + ab, k);
    }
    std::complex<double> phi2k = wallk(10, k);

    std::complex<double> phis(0.);
    for (size_t a = 0; a < 4; a++) {
      phis += n[a] * phi[a];
    }
    for (size_t a = 0; a < 4; a++) {
      phir[a] = phi[a] - n[a] * phis;
    }

    A = contract_rho(n, rho);
    V = contract_rho(n, dualize_rho(rho));

    wallk_rotated(0, k) = phis;
    for (size_t a = 0; a < 4; a++) {
      wallk_rotated(1 + a, k) = phir[a];
      wallk_rotated(5 + a, k) = A[a];
      wallk_rotated(9 + a, k) = V[a];
    }
    wallk_rotated(13, k) = phi2k;
  }
}

void Measurer::computeDerivedObs() {
  // NB: the intent is that this is to be called only
  // from the rank=0

  // Initializing the average to zero
  OAverage = std::vector<PetscScalar>(NScalars, 0.);

  // Compute the spatial correlation function of wall averages
  for (int l = 0; l < NObs; l++) {
    for (int i = 0; i < N; i++) {
      OAverage[l] += wallX(l, i);
    }
  }
  // Compute <X>
  for (int l = 0; l < NObs; l++) {
    OAverage[l] /= PetscReal(N);
  }

  fftw->execute(wallX, wallX_k);
  fftw->execute(wallY, wallY_k);
  fftw->execute(wallZ, wallZ_k);

  rotate_to_vev(wallX_k, wallX_k_rotated);
  rotate_to_vev(wallY_k, wallY_k_rotated);
  rotate_to_vev(wallZ_k, wallZ_k_rotated);

  fftw->execute(wallXPhase, wallXPhase_k);
  fftw->execute(wallYPhase, wallYPhase_k);
  fftw->execute(wallZPhase, wallZPhase_k);
}

/////////////////////////////////////////////////////////////////////////
//! Helper routine which opens the ASCII viewer with a
//! specific mode
PetscErrorCode PetscViewerASCIIOpenMode(MPI_Comm comm, const char *name,
                                        PetscFileMode mode,
                                        PetscViewer *viewer) {
  PetscViewerCreate(PETSC_COMM_SELF, viewer);
  PetscViewerSetType(*viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(*viewer, mode);
  PetscViewerFileSetName(*viewer, name);
  return (0);
}

measurer_output_txt::measurer_output_txt(Measurer *in) : measure(in) {

  const auto &ahandler = measure->model->data.ahandler;
  std::string name(ahandler.outputfiletag + "_averages.txt");
  PetscInt ierr = PetscViewerASCIIOpenMode(
      PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE, &averages_asciiviewer);
  CHKERRV(ierr);
}

measurer_output_txt::~measurer_output_txt() {
  PetscViewerDestroy(&averages_asciiviewer);
}

void measurer_output_txt::save(const std::string &what) {

  if (what == "phi") {
    for (int iO = 0; iO < measure->NScalars; iO++) {
      PetscViewerASCIIPrintf(averages_asciiviewer, "%22.15e ",
                             measure->OAverage[iO]);
    }
    PetscViewerASCIIPrintf(averages_asciiviewer, "\n");
  }
}
