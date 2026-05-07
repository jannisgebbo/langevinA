#include "measurer.h"
#include "nvector.h"
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

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

  // Restore the array
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

// takes the solution, evolves it with diffusion stepper and locally decomposes solution
// along the diffused solution; repeats for each coarsen level 1..ncoarsen_steps
void Measurer::computeSliceAverageCoarsened(Vec *solution) {

  // Get the local information
  DM &da = model->domain;
  const auto &data = model->data;

  Vec localU, localU_coarsened;
  G_node ***fld, ***fld_coarsened;
  // localU has the dimensions of the local domain
  DMGetLocalVector(da, &localU);
  DMGetLocalVector(da, &localU_coarsened);
  // take the global solution and distribute to the local vector localU
  DMGlobalToLocalBegin(da, *solution, INSERT_VALUES, localU);
  DMGlobalToLocalEnd(da, *solution, INSERT_VALUES, localU);
  // From the vector define the pointer for the field phi
  DMDAVecGetArrayRead(da, localU, &fld);

  // deep copy solution into solution_coarsened; will be evolved one step at a time
  VecCopy(*solution, solution_coarsened);

  // Get the ranges (same for all coarsen levels)
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

  // Iterate through selected coarsen levels, applying incremental steps between them
  int current_step = 0;
  for (int c = 0; c < static_cast<int>(coarsen_levels.size()); c++) {
    int steps_to_apply = coarsen_levels[c] - current_step;
    // Apply the remaining steps to reach the next selected level
    diffuser->step_coarsening(data.atime.dt(), &solution_coarsened, steps_to_apply);
    current_step = coarsen_levels[c];

    // convert solution_coarsened to a local 3d array
    DMGlobalToLocalBegin(da, solution_coarsened, INSERT_VALUES, localU_coarsened);
    DMGlobalToLocalEnd(da, solution_coarsened, INSERT_VALUES, localU_coarsened);
    DMDAVecGetArrayRead(da, localU_coarsened, &fld_coarsened);

    // Set up the slice averages initialized to zero
    std::fill(wallXCoarse[c].v.begin(), wallXCoarse[c].v.end(), 0.);
    std::fill(wallYCoarse[c].v.begin(), wallYCoarse[c].v.end(), 0.);
    std::fill(wallZCoarse[c].v.begin(), wallZCoarse[c].v.end(), 0.);

    // Local arrays with same dimensions initialized to zero
    nvector<double, 2> wallXCoarseLocal(NObsCoarse, N);
    nvector<double, 2> wallYCoarseLocal(NObsCoarse, N);
    nvector<double, 2> wallZCoarseLocal(NObsCoarse, N);

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

          // Gram-Schmidt procedure:
          // phi_project = phi - <phi, phi_coarse>/<phi_coarse, phi_coarse> phi_coarse

          double norm2_c = 0.0;   // <phi_coarse, phi_coarse>
          double norm2 = 0.0;     // <phi, phi>
          double phi_s = 0.0;     // <phi, phi_coarse>
          for (int l=0; l < ModelAData::Nphi; l++){
            // unnormalized coarse spin
            n[l] = fld_coarsened[k][j][i].f[l];
            // solution spin
            phi[l] = fld[k][j][i].f[l];
            // scalar product
            phi_s += n[l]*phi[l];
            // norm_c
            norm2_c += pow(n[l], 2);
            // norm2
            norm2 += pow(phi[l], 2);
          }
          // sigma is norm of projected which is
          // norm = sqrt(<phi, phi> - 2<phi_coarse, phi>^2/<phi_coarse, phi_coarse>
          //             + <phi, phi_coarse>^2 / <phi_coarse, phi_coarse> )
          //      = sqrt(norm2 - phi_s^2/norm2_c )
          sigma = sqrt(norm2 - phi_s*phi_s/norm2_c );
          // store charge fields
          for (int l = 0; l < ModelAData::NA; l++) {
            rho[l] = fld[k][j][i].A[l];
            rho[l + ModelAData::NA] = fld[k][j][i].V[l];
          }
          // project phi
          for (int l = 0; l < ModelAData::Nphi; l++) {
            // project phi
            //phi[l] = phi[l] - phi_s/norm2_c * n[l];
            // normalize n to unity
            n[l] = n[l] / sqrt(norm2_c);
          }
          // project charge fields
          A = contract_rho(n, rho);
          V = contract_rho(n, dualize_rho(rho));

          // store projected in wall*CoarseLocal
          wallXCoarseLocal(0, i) += sigma;
          wallYCoarseLocal(0, j) += sigma;
          wallZCoarseLocal(0, k) += sigma;
          for (int l = 0; l < ModelAData::Nphi; l++) {
            wallXCoarseLocal(1 + l, i) += phi[l];
            wallYCoarseLocal(1 + l, j) += phi[l];
            wallZCoarseLocal(1 + l, k) += phi[l];

            wallXCoarseLocal(5 + l, i) += A[l];
            wallYCoarseLocal(5 + l, j) += A[l];
            wallZCoarseLocal(5 + l, k) += A[l];

            wallXCoarseLocal(9 + l, i) += V[l];
            wallYCoarseLocal(9 + l, j) += V[l];
            wallZCoarseLocal(9 + l, k) += V[l];
          }
          wallXCoarseLocal(Measurer::NObsCoarse - 1, i) += sigma * sigma;
          wallYCoarseLocal(Measurer::NObsCoarse - 1, j) += sigma * sigma;
          wallZCoarseLocal(Measurer::NObsCoarse - 1, k) += sigma * sigma;
        }
      }
    }

    // normalize by 1/N^2
    for (int l = 0; l < NObsCoarse; ++l) {
      for (int i = 0; i < N; ++i) {
        wallXCoarseLocal(l, i) /= PetscReal(N * N);
        wallYCoarseLocal(l, i) /= PetscReal(N * N);
        wallZCoarseLocal(l, i) /= PetscReal(N * N);
      }
    }

    DMDAVecRestoreArrayRead(da, localU_coarsened, &fld_coarsened);

    // Bring all the data into one
    for (int l = 0; l < NObsCoarse; l++) {
      MPI_Reduce(&wallXCoarseLocal(l, 0), &wallXCoarse[c](l, 0), N, MPIU_SCALAR,
                 MPI_SUM, 0, PETSC_COMM_WORLD);
      MPI_Reduce(&wallYCoarseLocal(l, 0), &wallYCoarse[c](l, 0), N, MPIU_SCALAR,
                 MPI_SUM, 0, PETSC_COMM_WORLD);
      MPI_Reduce(&wallZCoarseLocal(l, 0), &wallZCoarse[c](l, 0), N, MPIU_SCALAR,
                 MPI_SUM, 0, PETSC_COMM_WORLD);
    }
  }

  // Restore the uncoarsened arrays
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);
  DMRestoreLocalVector(da, &localU_coarsened);
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

  for (int c = 0; c < static_cast<int>(coarsen_levels.size()); c++) {
    fftw->execute(wallXCoarse[c], wallXCoarse_k[c]);
    fftw->execute(wallYCoarse[c], wallYCoarse_k[c]);
    fftw->execute(wallZCoarse[c], wallZCoarse_k[c]);
  }
}

// Routine: compute different terms that contribute to total energy and 
// store inside Energy[NEnergy]
// 
// H = H_s + H_A + H_V + H_H
// H: total energy - Energy[0]
// H_s: spin field gradient term - Energy[1]
// H_A: axial charge field term - Energy[2]
// H_V: vector charge field term - Energy[3]
// H_H: term induced by presence of magnetic field - Energy[4]
void Measurer::computeEnergy() {
  DM da = model->domain;
  // Get a local vector with ghost cells
  Vec localUNew;
  DMGetLocalVector(da, &localUNew);

  // Fill in the ghost celss with mpicalls

  DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localUNew);
  DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localUNew);

  const auto &data = model->data;
  const auto &coeff = data.acoefficients;

  G_node ***phiNew;
  DMDAVecGetArrayRead(da, localUNew, &phiNew);

  const PetscReal H[4] = {coeff.sigmabyf(model->data.atime.t())*coeff.H, 0., 0., 0.};

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // Loop over central elements
  PetscScalar phimid = 0, grad2 = 0, nA2 = 0, nV2 = 0, hEn = 0;

  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        for (int s = 0; s < ModelAData::Nphi; ++s) {

          phimid = phiNew[k][j][i].f[s];

          grad2 += pow(phiNew[k + 1][j][i].f[s] - phimid, 2);
          grad2 += pow(phiNew[k][j + 1][i].f[s] - phimid, 2);
          grad2 += pow(phiNew[k][j][i + 1].f[s] - phimid, 2);

          hEn += phimid * H[s];
        }
        for (PetscInt s = 0; s < ModelAData::NV; s++) {
          nV2 += pow(phiNew[k][j][i].V[s], 2);
        }

        for (PetscInt s = 0; s < ModelAData::NA; s++) {
          nA2 += pow(phiNew[k][j][i].A[s], 2);
        }
      }
    }
  }

  Energy = std::vector<PetscScalar>(NEnergy, 0.);
  std::vector<PetscScalar> EnergyLocal(NEnergy, 0.);
    
  EnergyLocal[1] = 0.5 / pow(PetscReal(N),3) * grad2;
  EnergyLocal[2] = 0.5 / pow(PetscReal(N),3) / coeff.chi * nA2;
  EnergyLocal[3] = 0.5 / pow(PetscReal(N),3) / coeff.chi * nV2;
  EnergyLocal[4] = -1.0 / pow(PetscReal(N),3) * hEn;

  for (int l = 1; l < NEnergy; l++) {
     EnergyLocal[0] += EnergyLocal[l];
  }

  MPI_Reduce(&EnergyLocal[0], &Energy[0], NEnergy, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);

  DMDAVecRestoreArrayRead(da, localUNew, &phiNew);
  DMRestoreLocalVector(da, &localUNew);

}

// Routine: compute different terms that contribute to total energy and 
// store inside EnergyRotated[NEnergy]
// 
// H = H_s + H_A + H_V + H_H
// H: total energy - Energy[0]
// H_s: pion field gradient term - Energy[1]
// H_A: rotated axial charge field term - Energy[2]
// H_V: rotated vector charge field term - Energy[3]
// H_m2: pion field mass term - Energy[4]
void Measurer::computeEnergyRotated() {
  DM da = model->domain;
  // Get a local vector with ghost cells
  Vec localUNew;
  DMGetLocalVector(da, &localUNew);

  // Fill in the ghost celss with mpicalls

  DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localUNew);
  DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localUNew);

  const auto &data = model->data;
  const auto &coeff = data.acoefficients;

  G_node ***phiNew;
  DMDAVecGetArrayRead(da, localUNew, &phiNew);

  const PetscReal m2 = coeff.sigmabyf(data.atime.t())*coeff.H/sqrt(coeff.f2(data.atime.t()));

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // rotated fields
  std::array<PetscScalar, ModelAData::Nphi> n{};
  std::array<PetscScalar, ModelAData::Nphi> phi{};
  std::array<PetscScalar, ModelAData::Nphi> phir{};
  std::array<PetscScalar, ModelAData::Nphi> phi_xplus{};
  std::array<PetscScalar, ModelAData::Nphi> phir_xplus{};
  std::array<PetscScalar, ModelAData::Nphi> phi_yplus{};
  std::array<PetscScalar, ModelAData::Nphi> phir_yplus{};
  std::array<PetscScalar, ModelAData::Nphi> phi_zplus{};
  std::array<PetscScalar, ModelAData::Nphi> phir_zplus{};
  std::array<PetscScalar, 6> rho{};
  std::array<PetscScalar, 4> V{};
  std::array<PetscScalar, 4> A{};

  // Loop over central elements
  PetscScalar phimid = 0, grad2 = 0, nA2 = 0, nV2 = 0, hEn = 0;

  // Extract the vev direction
  double norm = 0.; 
  for (int a = 0; a < 4; a++) {
    n[a] = wallX_k(a, 0).real(); // Extract the zero mode  of phi_a
    norm += std::norm(n[a]);
  }
  norm = sqrt(norm);
  if (norm < std::numeric_limits<double>::min()) {
    n = std::array<PetscScalar, 4>{1, 0, 0, 0};
  } 
  else {
    for (size_t a = 0; a < 4; a++) {
      n[a] /= norm;
    }
  }
      
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        // Do the rotation to vev
        for (PetscInt s = 0; s < ModelAData::Nphi; s++) {
          phi[s] = phiNew[k][j][i].f[s];
          phi_xplus[s] = phiNew[k][j][i+1].f[s];
          phi_yplus[s] = phiNew[k][j+1][i].f[s];
          phi_zplus[s] = phiNew[k+1][j][i].f[s];
        }
        for (PetscInt s = 0; s < ModelAData::NA; s++) {
          rho[s] = phiNew[k][j][i].A[s];
        }
        for (PetscInt s = 0; s < ModelAData::NV; s++) {
          rho[s + ModelAData::NA] = phiNew[k][j][i].V[s];
        }

        PetscScalar phis(0.);
        PetscScalar phis_xplus(0.);
        PetscScalar phis_yplus(0.);
        PetscScalar phis_zplus(0.);
        for (size_t a = 0; a < 4; a++) {
          phis += n[a] * phi[a];
          phis_xplus += n[a] * phi_xplus[a];
          phis_yplus += n[a] * phi_yplus[a];
          phis_zplus += n[a] * phi_zplus[a];  
        }
        for (size_t a = 0; a < 4; a++) {
          phir[a] = phi[a] - n[a] * phis;
          phir_xplus[a] = phi_xplus[a] - n[a] * phis_xplus;
          phir_yplus[a] = phi_yplus[a] - n[a] * phis_yplus;  
          phir_zplus[a] = phi_zplus[a] - n[a] * phis_zplus;  
        }
        A = contract_rho(n, rho);
        V = contract_rho(n, dualize_rho(rho));
        
        for (int s = 1; s < ModelAData::Nphi; s++) {

          phimid = phir[s];

          grad2 += pow(phi_zplus[s] - phimid, 2);
          grad2 += pow(phi_yplus[s] - phimid, 2);
          grad2 += pow(phi_xplus[s] - phimid, 2);

          if (s!=0){  
            grad2 += m2 * pow(phimid, 2);
          }    
        }
        //hEn -= phi[0];
        for (PetscInt s = 0; s < 4; s++) {
          nV2 += pow(V[s], 2);
        }

        for (PetscInt s = 0; s < 4; s++) {
          nA2 += pow(A[s], 2);
        }
      }
    }
  }

  EnergyRotated = std::vector<PetscScalar>(NEnergy, 0.);
  std::vector<PetscScalar> EnergyLocal(NEnergy, 0.);
    
  EnergyLocal[1] = 0.5 / pow(PetscReal(N),3) * grad2;
  EnergyLocal[2] = 0.5 / pow(PetscReal(N),3) / coeff.chi * nA2;
  EnergyLocal[3] = 0.5 / pow(PetscReal(N),3) / coeff.chi * nV2;
  EnergyLocal[4] = coeff.sigmabyf(model->data.atime.t())*coeff.H / pow(PetscReal(N),3) * hEn;

  for (int l = 1; l < NEnergy; l++) {
     EnergyLocal[0] += EnergyLocal[l];
  }

  MPI_Reduce(&EnergyLocal[0], &EnergyRotated[0], NEnergy, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);

  DMDAVecRestoreArrayRead(da, localUNew, &phiNew);
  DMRestoreLocalVector(da, &localUNew);

}

// Routine: compute different terms that contribute to total energy and 
// store inside EnergyPhase[NEnergy]
// 
// H = H_s + H_A + H_V + H_H
// H: total energy - Energy[0]
// H_s: phase field gradient term - Energy[1]
// H_A: phase axial charge field term - Energy[2]
// H_V: phase vector charge field term - Energy[3]
// H_m2: phase field mass term - Energy[4]
void Measurer::computeEnergyPhase() {
  DM da = model->domain;
  // Get a local vector with ghost cells
  Vec localUNew;
  DMGetLocalVector(da, &localUNew);

  // Fill in the ghost celss with mpicalls

  DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localUNew);
  DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localUNew);

  const auto &data = model->data;
  const auto &coeff = data.acoefficients;

  G_node ***phiNew;
  DMDAVecGetArrayRead(da, localUNew, &phiNew);

  const PetscReal m2 = coeff.sigmabyf(data.atime.t())*coeff.H/sqrt(coeff.f2(data.atime.t()));
    
  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // rotated fields
  std::array<PetscScalar, ModelAData::Nphi> n{};
  std::array<PetscScalar, ModelAData::Nphi> phi{};
  std::array<PetscScalar, 6> rho{};
  std::array<PetscScalar, 4> V{};
  std::array<PetscScalar, 4> A{};

  // Loop over central elements
  PetscScalar phimid = 0, grad2 = 0, grad2_0 = 0, nA2 = 0, nV2 = 0, hEn = 0, hEn_0 = 0;
      
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        double norm = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          phi[l] = phiNew[k][j][i].f[l];
          norm += pow(phi[l], 2);
        }
        norm = sqrt(norm);

        if (norm > 100. * std::numeric_limits<double>::min()) {
          for (int l = 0; l < ModelAData::Nphi; l++) {
            n[l] = phi[l] / norm;
          }
        } else {
          n = {1, 0, 0, 0};
        }

        for (int l = 0; l < ModelAData::NA; l++) {
          rho[l] = phiNew[k][j][i].A[l];
          rho[l + ModelAData::NA] = phiNew[k][j][i].V[l];
        }
        A = contract_rho(n, rho);
        V = contract_rho(n, dualize_rho(rho));

        
        for (int s = 0; s < ModelAData::Nphi; s++) {

          phimid = phiNew[k][j][i].f[s];

          if (s==0){
            grad2_0 += pow(phiNew[k + 1][j][i].f[s] - phimid, 2);
            grad2_0 += pow(phiNew[k][j + 1][i].f[s] - phimid, 2);
            grad2_0 += pow(phiNew[k][j][i + 1].f[s] - phimid, 2);
            //hEn_0 += pow(phimid, 2);  
          }  
          else {
            grad2 += pow(phiNew[k + 1][j][i].f[s] - phimid, 2);
            grad2 += pow(phiNew[k][j + 1][i].f[s] - phimid, 2);
            grad2 += pow(phiNew[k][j][i + 1].f[s] - phimid, 2);
            hEn += pow(phimid, 2);  
          }    
        }
        for (PetscInt s = 0; s < 4; s++) {
          nV2 += pow(V[s], 2);
        }

        for (PetscInt s = 0; s < 4; s++) {
          nA2 += pow(A[s], 2);
        }
      }
    }
  }

  EnergyPhase = std::vector<PetscScalar>(NEnergy, 0.);
  std::vector<PetscScalar> EnergyLocal(NEnergy, 0.);
    
  EnergyLocal[1] = 0.5 / pow(PetscReal(N),3) * (grad2 + m2 * hEn);
  EnergyLocal[2] = 0.5 / pow(PetscReal(N),3) / coeff.chi * nA2;
  EnergyLocal[3] = 0.5 / pow(PetscReal(N),3) / coeff.chi * nV2;
  EnergyLocal[4] =  0.5 / pow(PetscReal(N),3) * (grad2_0); // + m2 * hEn_0);

  for (int l = 1; l < NEnergy; l++) {
     EnergyLocal[0] += EnergyLocal[l];
  }

  MPI_Reduce(&EnergyLocal[0], &EnergyPhase[0], NEnergy, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);

  DMDAVecRestoreArrayRead(da, localUNew, &phiNew);
  DMRestoreLocalVector(da, &localUNew);

}
