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

// Input: solution vector
//
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
//   DM &da = model->domain;
//   const auto &coeff = model->data.acoefficients;
//   Vec localU;
//   DMGetLocalVector(da, &localU);
//   // take the global vector U and distribute to the local vector localU
//   DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localU);
//   DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localU);
// 
//   // From the vector define the pointer for the field phi
//   G_node ***fld;
//   DMDAVecGetArrayRead(da, localU, &fld);
// 
//   // Initializing the energy array to zero
//   Energy = std::vector<PetscScalar>(NEnergy, 0.);
//   // create local energy arrays
//   std::vector<PetscScalar> EnergyLocal(NEnergy, 0.);
// 
//   // Get the ranges
//   PetscInt ixs, iys, izs, nx, ny, nz;
//   DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
// 
//   int actualInd = 0;
//   int kplus1, jplus1, iplus1;
// 
//   // Store the local averages
//   for (int k = izs; k < izs + nz; k++) {
//     // take care of periodic boundary conditions
//     /*
//     if (k == (N-1)){
//       kplus1 = 0;
//     }
//     else {
//       kplus1 = k + 1;
//     }
//     */
//       kplus1 = k + 1;
//     for (int j = iys; j < iys + ny; j++) {
//       // take care of periodic boundary conditions
//     /*
//       if (j == (N-1)){
//         jplus1 = 0;
//       }
//       else {
//         jplus1 = j + 1;
//       }
//     */
//         jplus1 = j + 1;
//       for (int i = ixs; i < ixs + nx; i++) {
//         // take care of periodic boundary conditions
//     /*
//         if (i == (N-1)){
//           iplus1 = 0;
//         }
//         else {
//           iplus1 = i + 1;
//         }
//     */
//           iplus1 = i + 1;
// 
//         // add local part to different energy contributions
//         for (int l = 0; l < ModelAData::Nphi; l++) {
//           // field gradient 
//           EnergyLocal[1] += 0.5 * (pow(fld[kplus1][j][i].f[l],2) + pow(fld[k][jplus1][i].f[l],2) 
//               + pow(fld[k][j][iplus1].f[l],2) + 2.0*fld[k][j][i].f[l] * (
//                   3.0/2.0 * fld[k][j][i].f[l] - fld[kplus1][j][i].f[l] - fld[k][jplus1][i].f[l]
//                   - fld[k][j][iplus1].f[l] )); 
//         }
//         for (int l = ModelAData::Nphi; l < ModelAData::Nphi + ModelAData::NA;
//              l++) {
//           actualInd = l - ModelAData::Nphi;
//           // axial charge 
//           EnergyLocal[2] += 1/(2.0*coeff.chi) * pow(fld[k][j][i].A[actualInd],2); 
//         }
//         for (int l = ModelAData::Nphi + ModelAData::NA;
//              l < ModelAData::Nphi + ModelAData::NA + ModelAData::NV; l++) {
//           actualInd = l - ModelAData::Nphi - ModelAData::NA;
//           // vector charge 
//           EnergyLocal[3] += 1/(2.0*coeff.chi) * pow(fld[k][j][i].V[actualInd],2); 
//         }
//         // magnetic field - field  
//         EnergyLocal[4] += -1.0*coeff.H * coeff.sigmabyf(model->data.atime.t()) * fld[k][j][i].f[0]; 
//       }
//     }
//   }
//   // Dividing by volume and adding parts to total energy 
//   for (int l = 1; l < NEnergy; l++) {
//     EnergyLocal[l] /= pow(PetscReal(N),3);
//     EnergyLocal[0] += EnergyLocal[l];
//   }
//  
//   MPI_Reduce(&EnergyLocal[0], &Energy[0], NEnergy, MPIU_SCALAR, MPI_SUM, 0,
//              PETSC_COMM_WORLD);
//   // Retstore the array
//   DMDAVecRestoreArrayRead(da, localU, &fld);
//   DMRestoreLocalVector(da, &localU);


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

  const PetscReal H[4] = {coeff.H, 0., 0., 0.};

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // Loop over central elements
  PetscScalar phimid = 0, grad2 = 0, nab2 = 0, hEn = 0;
  PetscScalar localEnergy = 0;
  // std::array<PetscScalar,3> localEnergyArr {0,0,0};

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
          nab2 += pow(phiNew[k][j][i].V[s], 2);
        }

        for (PetscInt s = 0; s < ModelAData::NA; s++) {
          nab2 += pow(phiNew[k][j][i].A[s], 2);
        }
      }
    }
  }

  localEnergy += 0.5 / coeff.chi * nab2;
  localEnergy += 0.5 * grad2;
  localEnergy -= hEn;

  Energy = 0.0;
  MPI_Reduce(&localEnergy, &Energy, 1, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);

  DMDAVecRestoreArrayRead(da, localUNew, &phiNew);
  DMRestoreLocalVector(da, &localUNew);

}
