#include "Stepper.h"
#include "ModelA.h"
#include "O4AlgebraHelper.h"
#include <cstdio>


IdealLF::IdealLF(ModelA &in)
    : model(&in) {
}


bool IdealLF::step(const double &dt) {

  DM da = model->domain;
  // Get a local vector with ghost cells
  Vec localU;
  DMGetLocalVector(da, &localU);

  // Fill in the ghost celss with mpicalls
  DMGlobalToLocalBegin(da, model->previoussolution, INSERT_VALUES, localU);
  DMGlobalToLocalEnd(da, model->previoussolution, INSERT_VALUES, localU);

  const ModelAData &data = model->data;

  G_node ***phi;
  DMDAVecGetArrayRead(da, localU, &phi);

  G_node ***phinew;
  DMDAVecGetArray(da, model->solution, &phinew);

  const PetscReal H[4] = {data.H, 0., 0., 0.};
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);

  PetscScalar advxx, advyy, advzz;
  PetscInt s1, s2, epsilon;
  PetscInt  xstart, ystart, zstart, xdimension, ydimension,
      zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // Loop over central elements
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        // First evolve the momenta nab

        G_node &centralPhi = phi[k][j][i];
        G_node &phixplus = phi[k][j][i + 1];
        G_node &phixminus = phi[k][j][i - 1];
        G_node &phiyplus = phi[k][j + 1][i];
        G_node &phiyminus = phi[k][j - 1][i];
        G_node &phizplus = phi[k + 1][j][i];
        G_node &phizminus = phi[k - 1][j][i];

        for (PetscInt l = 0; l < ModelAData::NA; l++) {
          advxx = (-phixplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phixplus.f[l + 1] +
                   centralPhi.f[0] * phixminus.f[l + 1] -
                   phixminus.f[0] * centralPhi.f[l + 1]) *
                  axx;

          advyy = (-phiyplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phiyplus.f[l + 1] +
                   centralPhi.f[0] * phiyminus.f[l + 1] -
                   phiyminus.f[0] * centralPhi.f[l + 1]) *
                  ayy;

          advzz = (-phizplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phizplus.f[l + 1] +
                   centralPhi.f[0] * phizminus.f[l + 1] -
                   phizminus.f[0] * centralPhi.f[l + 1]) *
                  azz;

          phinew[k][j][i].A[l] += dt * (advxx + advyy + advzz - H[0] * centralPhi.f[l + 1]);
        }

        for (PetscInt s = 0; s < ModelAData::NV; s++) {

          s1 = (s + 1) % 3;
          s2 = (s + 2) % 3;
          epsilon = ((PetscScalar)(s - s1) * (s1 - s2) * (s2 - s)) / 2.;
          // advection term with epsilon
          advxx = epsilon *
                  (phixplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phixminus.f[1 + s1] -
                   phixplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phixminus.f[1 + s2]) *
                  axx;

          advyy = epsilon *
                  (phiyplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phiyminus.f[1 + s1] -
                   phiyplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phiyminus.f[1 + s2]) *
                  ayy;

          advzz = epsilon *
                  (phizplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phizminus.f[1 + s1] -
                   phizplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phizminus.f[1 + s2]) *
                  azz;

          phinew[k][j][i].V[s] += dt * (advxx + advyy + advzz);
        }

      	 // here we have to convert the charge from the chemical potential
      	PetscScalar axialmu[ModelAData::NA], vectormu[ModelAData::NV];

      	for(PetscInt s=0;s < ModelAData::NV; s++ ){

      		vectormu[s] = - phinew[k][j][i].V[s] / data.chi;
      	}

      	for(PetscInt s=0;s < ModelAData::NA; s++ ){

      		axialmu[s]= - phinew[k][j][i].A[s] / data.chi;


      	}

	// Then we rotate phi by n.

    //   O4AlgebraHelper::O4Rotation(phinew[k][j][i].V, phinew[k][j][i].A,phinew[k][j][i].f);

          O4AlgebraHelper::O4Rotation(vectormu, axialmu, phinew[k][j][i].f);

      }
    }
  }

  DMDAVecRestoreArrayRead(da, localU, &phi);
  DMRestoreLocalVector(da, &localU);

  DMDAVecRestoreArray(da, model->solution, &phinew);


  return true;
}




///////////////////////////////////////////////////////////////////////////

void IdealLF::finalize() { }
///////////////////////////////////////////////////////////////////////////

ForwardEuler::ForwardEuler(ModelA &in, bool wnoise)
    : model(&in), withNoise(wnoise) {
  VecDuplicate(model->solution, &noise);
}
void ForwardEuler::finalize() { VecDestroy(&noise); }

bool ForwardEuler::step(const double &dt) {

  PetscLogEvent random,  loop, create;
  PetscLogEventRegister("FillRandomVec", 0, &random);
  PetscLogEventRegister("PotentialEval", 0, &loop);
  PetscLogEventRegister("CreationAndMPI", 0, &create);

  /////////////////////////////////////////////////////////////////////////
  PetscLogEventBegin(random, 0, 0, 0, 0);
  if (withNoise) {
    ModelARndm->fillVec(noise);
  } else {
    VecZeroEntries(noise);
  }
  PetscLogEventEnd(random, 0, 0, 0, 0);
  /////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////
  PetscLogEventBegin(create, 0, 0, 0, 0);
  DM da = model->domain;
  // Get a local vector with ghost cells
  Vec localU;
  DMGetLocalVector(da, &localU);

  // Fill in the ghost celss with mpicalls
  DMGlobalToLocalBegin(da, model->previoussolution, INSERT_VALUES, localU);
  DMGlobalToLocalEnd(da, model->previoussolution, INSERT_VALUES, localU);

  G_node ***phi;
  DMDAVecGetArrayRead(da, localU, &phi);

  G_node ***gaussiannoise;
  DMDAVecGetArrayRead(da, noise, &gaussiannoise);

  G_node ***phinew;
  DMDAVecGetArray(da, model->solution, &phinew);

  PetscLogEventEnd(create, 0, 0, 0, 0);

  /////////////////////////////////////////////////////////////////////////

  PetscLogEventBegin(loop, 0, 0, 0, 0);

  // Get The local coordinates
  const ModelAData &data = model->data;
  PetscInt i, j, k, l, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // Parameters for loop
  const PetscReal dtg = data.gamma * dt;
  const PetscReal xdtg = sqrt(2. / dtg);
  const PetscReal &m2 = data.mass;
  const PetscReal &lambda = data.lambda;
  const PetscReal H[4] = {data.H, 0., 0., 0.};
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);
  G_node phidotI = {};

  // Loop over central elements
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {
        const PetscReal(&f)[4] = phi[k][j][i].f;
        const PetscReal(&xi)[4] = gaussiannoise[k][j][i].f;
        const PetscReal &phi2 =
            f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3];

        for (l = 0; l < ModelAData::Nphi; l++) {
          const PetscReal uxx =
              phi[k][j][i + 1].f[l] + phi[k][j][i - 1].f[l] - 2. * f[l];
          const PetscReal uyy =
              phi[k][j + 1][i].f[l] + phi[k][j - 1][i].f[l] - 2. * f[l];
          const PetscReal uzz =
              phi[k + 1][j][i].f[l] + phi[k - 1][j][i].f[l] - 2. * f[l];
          const PetscReal lap = axx * uxx + ayy * uyy + azz * uzz;

          phidotI.f[l] =
              lap - (m2 * f[l] + lambda * phi2 * f[l]) + H[l] + xdtg * xi[l];

          phinew[k][j][i].f[l] = f[l] + dtg * phidotI.f[l];
        }
      }
    }
  }
  PetscLogEventEnd(loop, 0, 0, 0, 0);

  /////////////////////////////////////////////////////////////////////////

  DMDAVecRestoreArray(da, model->solution, &phinew);

  DMDAVecRestoreArrayRead(da, noise, &gaussiannoise);
  DMDAVecRestoreArrayRead(da, localU, &phi);
  DMRestoreLocalVector(da, &localU);

  return true;
}

/////////////////////////////////////////////////////////////////////////

BackwardEuler::BackwardEuler(ModelA &in, bool wnoise)
    : model(&in), withNoise(wnoise) {

  VecDuplicate(model->solution, &auxsolution);
  VecDuplicate(model->solution, &noise);
  // Set the Jacobian matrix
  DMSetMatType(model->domain, MATAIJ);
  DMCreateMatrix(model->domain, &jacobian);

  SNESCreate(PETSC_COMM_WORLD, &Solver);
  SNESSetFunction(Solver, auxsolution, FormFunction, this);
  SNESSetJacobian(Solver, jacobian, jacobian, FormJacobian, this);
  SNESSetFromOptions(Solver);
}
void BackwardEuler::finalize() {
  MatDestroy(&jacobian);
  SNESDestroy(&Solver);
  VecDestroy(&noise);
  VecDestroy(&auxsolution);
}

bool BackwardEuler::step(const double &dt) {
  deltat = dt;
  if (withNoise) {
    ModelARndm->fillVec(noise);
  } else {
    VecZeroEntries(noise);
  }
  SNESSolve(Solver, NULL, model->solution);
  return true;
}

// Evaluate the rhs function for the nonlinear solver
PetscErrorCode BackwardEuler::FormFunction(SNES snes, Vec U, Vec F, void *ptr) {

  BackwardEuler *stepper = static_cast<BackwardEuler *>(ptr);
  ModelA *model = stepper->model;
  const ModelAData &data = model->data;
  PetscInt i, j, k, l, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;
  DM da = model->domain;

  // define a local vector with boundary cells
  Vec localU;
  DMGetLocalVector(da, &localU);

  // Get the Global dimension of the Grid
  // Define the spaceing
  PetscReal hx = data.hX();
  PetscReal hy = data.hY();
  PetscReal hz = data.hZ();

  // take the global vector U and distribute to the local vector localU
  DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU);
  DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU);

  // From the vector define the pointer for the field u and the rhs f
  G_node ***phi, ***f;
  DMDAVecGetArrayRead(da, localU, &phi);
  DMDAVecGetArray(da, F, &f);

  G_node ***gaussiannoise;
  DMDAVecGetArrayRead(da, stepper->noise, &gaussiannoise);

  G_node ***oldphi;
  DMDAVecGetArrayRead(da, model->previoussolution, &oldphi);

  // Get the local coordinates
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // This actually computes the right hand side
  PetscScalar uxx, uyy, uzz, ucentral, phisquare;
  G_node phidotI = {};
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {
        phisquare = 0.;
        for (l = 0; l < data.Nphi; l++) {
          phisquare = phisquare + phi[k][j][i].f[l] * phi[k][j][i].f[l];
        }
        for (l = 0; l < data.Nphi; l++) {
          ucentral = phi[k][j][i].f[l];
          uxx = (-2.0 * ucentral + phi[k][j][i - 1].f[l] +
                 phi[k][j][i + 1].f[l]) /
                (hx * hx);
          uyy = (-2.0 * ucentral + phi[k][j - 1][i].f[l] +
                 phi[k][j + 1][i].f[l]) /
                (hy * hy);
          uzz = (-2.0 * ucentral + phi[k - 1][j][i].f[l] +
                 phi[k + 1][j][i].f[l]) /
                (hz * hz);

          phidotI.f[l] =
              data.gamma * (uxx + uyy + uzz) -
              data.gamma * (data.mass + data.lambda * phisquare) * ucentral +
              (l == 0 ? data.gamma * data.H : 0.) +
              PetscSqrtReal(2. * data.gamma / stepper->deltat) *
                  gaussiannoise[k][j][i].f[l];

          // here you want to put the formula for the euler step F(phi)=0
          f[k][j][i].f[l] = -phi[k][j][i].f[l] + oldphi[k][j][i].f[l] +
                            stepper->deltat * phidotI.f[l];
        }
        // We are not updating the currents here
        for (l = 0; l < data.NA; l++) {
          f[k][j][i].A[l] = -phi[k][j][i].A[l] + oldphi[k][j][i].A[l];
        }
        for (l = 0; l < data.NV; l++) {
          f[k][j][i].V[l] = -phi[k][j][i].V[l] + oldphi[k][j][i].V[l];
        }
      }
    }
  }
  // We need to restore the vector in U and F
  DMDAVecRestoreArrayRead(da, localU, &phi);

  DMDAVecRestoreArray(da, F, &f);
  DMRestoreLocalVector(da, &localU);
  DMDAVecRestoreArrayRead(da, stepper->noise, &gaussiannoise);
  DMDAVecRestoreArrayRead(da, model->previoussolution, &oldphi);

  return (0);
}
// Form the Jacobian with Petsc Interface
PetscErrorCode BackwardEuler::FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre,
                                           void *ptr) {
  BackwardEuler *stepper = static_cast<BackwardEuler *>(ptr);
  ModelA *model = stepper->model;
  return FormJacobianGeneric(snes, U, J, Jpre, stepper->deltat, model);
}

// Form the Jacobian the Jabian generic interface
PetscErrorCode BackwardEuler::FormJacobianGeneric(SNES snes, Vec U, Mat J,
                                                  Mat Jpre, const double &dt,
                                                  ModelA *model) {

  DM da = model->domain;
  const ModelAData &data = model->data;
  DMDALocalInfo info;
  PetscInt i, j, k, l;

  // Get the local information and store in info
  DMDAGetLocalInfo(da, &info);
  Vec localU;
  DMGetLocalVector(da, &localU);
  // take the global vector U and distribute to the local vector localU
  DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU);
  DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU);

  // From the vector define the pointer for the field phi
  G_node ***phi;
  DMDAVecGetArrayRead(da, localU, &phi);

  // Define the spacing
  PetscReal hx = data.hX();
  PetscReal hy = data.hY();
  PetscReal hz = data.hZ();

  const double &gamma = data.gamma;
  for (k = info.zs; k < info.zs + info.zm; k++) {
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (i = info.xs; i < info.xs + info.xm; i++) {
        PetscScalar phisquare = 0.;
        for (l = 0; l < ModelAData::Nphi; l++) {
          phisquare = phisquare + phi[k][j][i].f[l] * phi[k][j][i].f[l];
        }
        for (l = 0; l < ModelAData::Nphi; l++) {
          // we define the column
          PetscInt nc = 0;
          MatStencil row, column[10];
          PetscScalar value[10];
          // here we insert the position of the row
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = l;
          // here we define de position of the non-vansih column for the given
          // row in total there are 7*4 entries and nc is the total number of
          // column per row x direction
          column[nc].i = i - 1;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = dt * gamma / (hx * hx);
          column[nc].i = i + 1;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = dt * gamma / (hx * hx);
          // y direction
          column[nc].i = i;
          column[nc].j = j - 1;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = dt * gamma / (hy * hy);
          column[nc].i = i;
          column[nc].j = j + 1;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = dt * gamma / (hy * hy);
          // z direction
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k - 1;
          column[nc].c = l;
          value[nc++] = dt * gamma / (hz * hz);
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k + 1;
          column[nc].c = l;
          value[nc++] = dt * gamma / (hz * hz);
          // The central element need a loop over the flavour index of the
          // column (is a full matrix in the flavour index )
          for (PetscInt m = 0; m < ModelAData::Nphi; m++) {
            if (m == l) {
              column[nc].i = i;
              column[nc].j = j;
              column[nc].k = k;
              column[nc].c = l;
              value[nc++] =
                  -1. + dt * data.gamma *
                            (-2.0 / (hy * hy) - 2.0 / (hx * hx) -
                             2.0 / (hz * hz) - data.mass -
                             data.lambda * (phisquare + 2. * phi[k][j][i].f[l] *
                                                            phi[k][j][i].f[l]));
            } else {
              column[nc].i = i;
              column[nc].j = j;
              column[nc].k = k;
              column[nc].c = m;
              value[nc++] =
                  dt * data.gamma *
                  (-2. * data.lambda * phi[k][j][i].f[l] * phi[k][j][i].f[m]);
            }
          }
          // here we set the matrix
          MatSetValuesStencil(Jpre, 1, &row, nc, column, value, INSERT_VALUES);
        }

        // Put the minus the identity for non-evolved components
        for (l = 0; l < ModelAData::NA; l++) {
          PetscInt nc = 0; // number of non-zero entries
          MatStencil row = {};
          MatStencil column[1] = {};
          PetscScalar value[1];
          // here we insert the position of the row
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = l + ModelAData::Nphi;
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = row.c;
          value[nc++] = -1.;

          // here we set the matrix
          MatSetValuesStencil(Jpre, 1, &row, nc, column, value, INSERT_VALUES);
        }

        // Put the minus the identity for non-evolved components
        for (l = 0; l < ModelAData::NV; l++) {
          PetscInt nc = 0; // number of non-zero entries
          MatStencil row = {};
          MatStencil column[1] = {};
          PetscScalar value[1];
          // here we insert the position of the row
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = l + ModelAData::Nphi + ModelAData::NA;
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = row.c;
          value[nc++] = -1.;

          // here we set the matrix
          MatSetValuesStencil(Jpre, 1, &row, nc, column, value, INSERT_VALUES);
        }
      }
    }
  }
  MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);
  if (J != Jpre) {
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
  }
  DMDAVecRestoreArrayRead(da, localU, &phi);
  DMRestoreLocalVector(da, &localU);

  return (0);
}

/////////////////////////////////////////////////////////////////////////

SemiImplicitBEuler::SemiImplicitBEuler(ModelA &in, bool wnoise)
    : model(&in), withNoise(wnoise) {
  VecDuplicate(model->solution, &rhs);
  VecDuplicate(model->solution, &noise);
  DMCreateMatrix(model->domain, &J);

  double hx = model->data.hX();
  double hy = model->data.hY();
  double hz = model->data.hZ();
  Form3PointLaplacian(model->domain, J, hx, hy, hz);

  MatConvert(J, MATSAME, MAT_INITIAL_MATRIX, &A);
  KSPCreate(PETSC_COMM_WORLD, &ksp);
}

void SemiImplicitBEuler::finalize() {
  KSPDestroy(&ksp);
  MatDestroy(&A);
  MatDestroy(&J);
  VecDestroy(&noise);
  VecDestroy(&rhs);
}

bool SemiImplicitBEuler::step(const double &dt) {

  if (withNoise) {
    ModelARndm->fillVec(noise);
  } else {
    VecZeroEntries(noise);
  }
  // Compute the RHS of the equation to be solved

  G_node ***bI;
  DMDAVecGetArray(model->domain, rhs, &bI);

  G_node ***phiI;
  DMDAVecGetArray(model->domain, model->solution, &phiI);

  G_node ***xiI;
  DMDAVecGetArray(model->domain, noise, &xiI);

  PetscInt i, j, k, l, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;

  DMDAGetCorners(model->domain, &xstart, &ystart, &zstart, &xdimension,
                 &ydimension, &zdimension);

  const ModelAData &data = model->data;
  double dtg = dt * data.gamma;
  double m2 = data.mass;
  double lambda = data.lambda;
  double H[ModelAData::Nphi] = {data.H, 0, 0, 0};

  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {

        G_node &phi = phiI[k][j][i];
        G_node &b = bI[k][j][i];
        G_node &xi = xiI[k][j][i];

        double phi2 = 0.;
        for (l = 0; l < ModelAData::Nphi; l++) {
          phi2 += phi.f[l] * phi.f[l];
        }

        for (l = 0; l < ModelAData::Nphi; l++) {
          b.f[l] = phi.f[l] / dtg - (m2 * phi.f[l] + lambda * phi2 * phi.f[l]) +
                   H[l] + xi.f[l] * sqrt(2. / dtg);
        }
        for (l = 0; l < ModelAData::NA; l++) {
          b.A[l] = phi.A[l] / dtg;
        }
        for (l = 0; l < ModelAData::NV; l++) {
          b.V[l] = phi.V[l] / dtg;
        }
      }
    }
  }
  DMDAVecRestoreArray(model->domain, noise, &xiI);
  DMDAVecRestoreArray(model->domain, model->solution, &phiI);
  DMDAVecRestoreArray(model->domain, rhs, &bI);

  // Construct the matrix for the equation to be solved
  MatCopy(J, A, SAME_NONZERO_PATTERN);
  MatShift(A, 1. / dtg);

  // Actually solve
  KSPSetOperators(ksp, A, A);
  KSPSetFromOptions(ksp);
  KSPSolve(ksp, rhs, model->solution);

  return true;
}

// Form the Jacobian the Jabian generic interface
PetscErrorCode SemiImplicitBEuler::Form3PointLaplacian(DM da, Mat J,
                                                       const double &hx,
                                                       const double &hy,
                                                       const double &hz) {
  // Get the local information and store in info
  DMDALocalInfo info;
  DMDAGetLocalInfo(da, &info);
  PetscInt i, j, k, l;
  for (k = info.zs; k < info.zs + info.zm; k++) {
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (i = info.xs; i < info.xs + info.xm; i++) {
        for (l = 0; l < ModelAData::Nphi; l++) {
          // we define the column
          PetscInt nc = 0;
          MatStencil row, column[10];
          PetscScalar value[10];
          // here we insert the position of the row
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = l;
          // here we define de position of the non-vansih column for the given
          // row in total there are 7*4 entries and nc is the total number of
          // column per row x direction
          column[nc].i = i - 1;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hx * hx);
          column[nc].i = i + 1;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hx * hx);
          // y direction
          column[nc].i = i;
          column[nc].j = j - 1;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hy * hy);
          column[nc].i = i;
          column[nc].j = j + 1;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hy * hy);
          // z direction
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k - 1;
          column[nc].c = l;
          value[nc++] = -1. / (hz * hz);
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k + 1;
          column[nc].c = l;
          value[nc++] = -1. / (hz * hz);

          // The central element need a loop over the flavour index of the
          // column (is a full matrix in the flavour index )
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz);

          // here we set the matrix
          MatSetValuesStencil(J, 1, &row, nc, column, value, INSERT_VALUES);
        }
      }
    }
  }
  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
  return (0);
}

/////////////////////////////////////////////////////////////////////////a

EulerLangevinHB::EulerLangevinHB(ModelA &in) : model(&in) {
  DMCreateLocalVector(model->domain, &phi_local);
}

bool EulerLangevinHB::step(const double &dt) {

  // Get pointer to local array
  G_node ***phi;
  DMDAVecGetArray(model->domain, phi_local, &phi);
  // Get pointer to local array
  G_node ***phinew;
  DMDAVecGetArray(model->domain, model->solution, &phinew);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(model->domain, &ixs, &iys, &izs, &nx, &ny, &nz);

  G_node heff = {};
  G_node phi_o = {};
  G_node phi_n = {};

  const double &Lambda2 = 0.5 * (2. * 3. + model->data.mass); // mass term
  const double &lambda = model->data.lambda;
  const double &H = model->data.H;
  const PetscReal rdtg = sqrt(2. * dt * model->data.gamma);

  // Checkerboard order ieo = even and odd sites
  for (int ieo = 0; ieo < 2; ieo++) {
    // take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(model->domain, model->solution, INSERT_VALUES,
                         phi_local);
    DMGlobalToLocalEnd(model->domain, model->solution, INSERT_VALUES,
                       phi_local);

    for (int k = izs; k < izs + nz; k++) {
      for (int j = iys; j < iys + ny; j++) {
        for (int i = ixs; i < ixs + nx; i++) {
          if ((k + j + i) % 2 != ieo) {
            continue;
          }

          PetscScalar hphi_o = 0.; // old and new h*phi
          PetscScalar hphi_n = 0.; // old and new h*phi
          PetscScalar s_o = 0.;    // old sum of phi**2
          PetscScalar s_n = 0.;    // new sum of phi**2

          for (int l = 0; l < ModelAData::Nphi; l++) {
            heff.f[l] = (phi[k][j][i + 1].f[l] + phi[k][j][i - 1].f[l]) +
                        (phi[k][j + 1][i].f[l] + phi[k][j - 1][i].f[l]) +
                        (phi[k + 1][j][i].f[l] + phi[k - 1][j][i].f[l]);

            phi_o.f[l] = phi[k][j][i].f[l];
            phi_n.f[l] = phi_o.f[l] + rdtg * ModelARndm->normal();

            s_o += pow(phi_o.f[l], 2);
            s_n += pow(phi_n.f[l], 2);

            hphi_o += heff.f[l] * phi_o.f[l];
            hphi_n += heff.f[l] * phi_n.f[l];
          }
          hphi_o += (H * phi_o.f[0]); // Take care of the true H
          hphi_n += (H * phi_n.f[0]);

          double dS = -(hphi_n - hphi_o) + Lambda2 * (s_n - s_o) +
                      lambda / 4. * (s_n * s_n - s_o * s_o);

          // Downward step
          if (dS < 0) {
            for (int l = 0; l < ModelAData::Nphi; l++) {
              phinew[k][j][i].f[l] = phi_n.f[l];
            }
            monitor.increment_down(dS);
            continue;
          }
          // Process upward step
          double r = ModelARndm->uniform();
          if (r < exp(-dS)) {
            // keep the upward step w. probl exp(-dS)
            for (int l = 0; l < ModelAData::Nphi; l++) {
              phinew[k][j][i].f[l] = phi_n.f[l];
            }
            monitor.increment_up_yes(dS);
            continue;
          } else {
            // dphi is rejected. Set the action change dS=0
            monitor.increment_up_no(0.);
            continue;
          }
        }
      }
    }
  }

  // Retstore the array
  DMDAVecRestoreArray(model->domain, phi_local, &phi);
  DMDAVecRestoreArray(model->domain, model->solution, &phinew);

  return true;
}
void EulerLangevinHB::finalize() {
  VecDestroy(&phi_local);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    monitor.print(stdout);
  }
}

/////////////////////////////////////////////////////////////////////////

// The six possible faces of the cube on the boundary of cell A and cell B. For
// instance,  face_cases[0] (the first row), is an even site site for cell B,
// eoA=0, cell B is shifted by plus one in the x direction.  and not shited in
// the y and z directions. face_cases[5] (the last row) is odd (eoA=1) for cell
// A, and cell B is shifted in z by one unit
/* clang-format off */
g_face_case g_face_cases[3][2] = {
    0, 1, 0, 0,
    1, 1, 0, 0,
    0, 0, 1, 0,
    1, 0, 1, 0,
    0, 0, 0, 1,
    1, 0, 0, 1,
};
/* clang-format on */

// Update a pair of cells A and B with the Heat Bath
double modelg_update_charge_pair(const double &chi, const double &rms,
                                 const double &nA, const double &nB,
                                 o4_stepper_monitor &monitor) {
  double q = rms * ModelARndm->normal();
  double dS =
      (pow(nA - q, 2) + pow(nB + q, 2) - pow(nA, 2) - pow(nB, 2)) / (2. * chi);
  // Downward step
  if (dS < 0) {
    monitor.increment_down(dS);
    return q;
  }
  // Process upward step
  double r = ModelARndm->uniform();
  if (r < exp(-dS)) {
    // keep the upward step w. probl exp(-dS)
    monitor.increment_up_yes(dS);
    return q;
  } else {
    // q is rejected. Set the action change dS=0
    monitor.increment_up_no(0.);
    return 0.;
  }
}

ModelGChargeHB::ModelGChargeHB(ModelA &in) : model(&in) {
  DMCreateLocalVector(model->domain, &phi_local);
  DMCreateLocalVector(model->domain, &dn_local);
}

bool ModelGChargeHB::step(const double &dt) {

  // Get pointer to local array
  data_node ***phi;
  DMDAVecGetArray(model->domain, phi_local, &phi);

  data_node ***dn;
  DMDAVecGetArray(model->domain, dn_local, &dn);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(model->domain, &ixs, &iys, &izs, &nx, &ny, &nz);

  const PetscReal rdtsigma = sqrt(2. * dt * model->data.sigma);
  const PetscReal chi = model->data.chi;

  // Shuffle the order of xyz and the order of even and odd
  // to eliminate potential bias.
  std::array<int, 3> orderxyz{0, 1, 2};
  std::array<int, 2> ordereo{0, 1};
  if (model->rank == 0) {
    shuffle(orderxyz.begin(), orderxyz.end(), ModelARndm->generator());
    shuffle(ordereo.begin(), ordereo.end(), ModelARndm->generator());
  }
  MPI_Bcast(orderxyz.data(), 3, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(ordereo.data(), 2, MPI_INT, 0, PETSC_COMM_WORLD);

  // Checkerboard order ieo = even and odd sites
  for (int ixyz = 0; ixyz < 3; ixyz++) {
    for (int ieo = 0; ieo < 2; ieo++) {
      // get the face case that we will update
      g_face_case &face = g_face_cases[orderxyz[ixyz]][ordereo[ieo]];

      // take the global vector U and distribute to the local vector localU
      DMGlobalToLocalBegin(model->domain, model->solution, INSERT_VALUES,
                           phi_local);
      DMGlobalToLocalEnd(model->domain, model->solution, INSERT_VALUES,
                         phi_local);

      // Zero out the differences
      VecSet(dn_local, 0.);
      for (int k = izs; k < izs + nz; k++) {
        for (int j = iys; j < iys + ny; j++) {
          for (int i = ixs; i < ixs + nx; i++) {
            if ((k + j + i) % 2 != face.eoA) {
              continue;
            }
            for (int L = ModelAData::Nphi; L < ModelAData::Ndof; L++) {
              int iB = i + face.iB;
              int jB = j + face.jB;
              int kB = k + face.kB;
              const PetscScalar &nA = phi[k][j][i].x[L];
              const PetscScalar &nB = phi[kB][jB][iB].x[L];
              PetscScalar q =
                  modelg_update_charge_pair(chi, rdtsigma, nA, nB, qmonitor);

              dn[k][j][i].x[L] = -q;
              dn[kB][jB][iB].x[L] = q;
            }
          }
        }
      }
      DMLocalToGlobal(model->domain, dn_local, ADD_VALUES, model->solution);
    }
  }
  return true;
};

void ModelGChargeHB::finalize() {
  VecDestroy(&phi_local);
  VecDestroy(&dn_local);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    fprintf(stdout, "Statistics of current updates.... \n");
    qmonitor.print(stdout);
  }
}

/////////////////////////////////////////////////////////////////////////

ModelGChargeCN::ModelGChargeCN(ModelA &in, bool wnoise)
    : model(&in), withNoise(wnoise) {

  VecDuplicate(model->solution, &rhs);
  VecDuplicate(model->solution, &dn);
  DMCreateLocalVector(model->domain, &noise_local);

  DMCreateMatrix(model->domain, &J);

  double hx = model->data.hX();
  double hy = model->data.hY();
  double hz = model->data.hZ();
  Form3PointLaplacian(model->domain, J, hx, hy, hz);

  MatConvert(J, MATSAME, MAT_INITIAL_MATRIX, &A);
  KSPCreate(PETSC_COMM_WORLD, &ksp);
}

void ModelGChargeCN::finalize() {
  KSPDestroy(&ksp);
  MatDestroy(&A);
  MatDestroy(&J);
  VecDestroy(&noise_local);
  VecDestroy(&dn);
  VecDestroy(&rhs);
}

// We are solving
//
// (1/dtD  + J) n_+ = n/dtD  +  1/D div.xi
//
// Here J = - nabla^2,  dtD = dt * D
bool ModelGChargeCN::step(const double &dt) {

  PetscScalar ****bI;
  DMDAVecGetArrayDOF(model->domain, rhs, &bI);

  PetscScalar ****phiI;
  DMDAVecGetArrayDOF(model->domain, model->solution, &phiI);

  PetscScalar ****xiI;
  DMDAVecGetArrayDOF(model->domain, noise_local, &xiI);

  // accumulates the divergence of xi
  PetscScalar ****dnI;
  DMDAVecGetArrayDOF(model->domain, dn, &dnI);

  PetscInt i, j, k, L, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;

  // Compute the random fluxes
  DMDAGetCorners(model->domain, &xstart, &ystart, &zstart, &xdimension,
                 &ydimension, &zdimension);

  // Parameters needed
  const PetscReal &dtD = dt * model->data.sigma / model->data.chi;
  ;
  const PetscReal &rxbyD = sqrt(2. * model->data.chi / dtD);

  // Compute the divergence of the noise
  VecSet(noise_local, 0.);
  VecSet(dn, 0.);
  if (withNoise) {
    for (int ixyz = 0; ixyz < 3; ixyz++) {
      for (int ieo = 0; ieo < 2; ieo++) {
        // get the face case that we will update
        g_face_case face = g_face_cases[ixyz][ieo];
        for (k = zstart; k < zstart + zdimension; k++) {
          for (j = ystart; j < ystart + ydimension; j++) {
            for (i = xstart; i < xstart + xdimension; i++) {
              if ((k + j + i) % 2 != face.eoA) {
                continue;
              }
              for (int L = ModelAData::Nphi; L < ModelAData::Ndof; L++) {

                int iB = i + face.iB;
                int jB = j + face.jB;
                int kB = k + face.kB;

                // Assumes that hx = hy = hz=1!
                PetscScalar q = rxbyD * ModelARndm->normal();
                xiI[k][j][i][L] -= q;
                xiI[kB][jB][iB][L] += q;
              }
            }
          }
        }
      }
    }
  }
  DMLocalToGlobal(model->domain, noise_local, ADD_VALUES, dn);

  // Compute the RHS of the equation to be solved
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {

        const PetscScalar *phi = phiI[k][j][i];
        const PetscScalar *q = dnI[k][j][i];
        PetscScalar *b = bI[k][j][i];

        for (L = 0; L < ModelAData::Nphi; L++) {
          b[L] = phi[L] / dtD;
        }
        for (L = ModelAData::Nphi; L < ModelAData::Ndof; L++) {
          b[L] = phi[L] / dtD + q[L];
        }
      }
    }
  }
  DMDAVecRestoreArrayDOF(model->domain, dn, &dnI);
  DMDAVecRestoreArrayDOF(model->domain, noise_local, &xiI);
  DMDAVecRestoreArrayDOF(model->domain, model->solution, &phiI);
  DMDAVecRestoreArrayDOF(model->domain, rhs, &bI);

  // Construct the matrix for the equation to be solved
  MatCopy(J, A, SAME_NONZERO_PATTERN);
  MatShift(A, 1. / dtD);

  // Actually solve
  KSPSetOperators(ksp, A, A);
  KSPSetFromOptions(ksp);
  KSPSolve(ksp, rhs, model->solution);

  return true;
}

// Form the Jacobian the Jabian generic interface
PetscErrorCode ModelGChargeCN::Form3PointLaplacian(DM da, Mat J,
                                                   const double &hx,
                                                   const double &hy,
                                                   const double &hz) {
  // Get the local information and store in info
  DMDALocalInfo info;
  DMDAGetLocalInfo(da, &info);
  PetscInt i, j, k, l;
  for (k = info.zs; k < info.zs + info.zm; k++) {
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (i = info.xs; i < info.xs + info.xm; i++) {
        for (l = 0; l < ModelAData::Nphi; l++) {
          PetscInt nc = 0;
          MatStencil row, column[10];
          PetscScalar value[10];
          // here we insert the position of the row
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = l;
          MatSetValuesStencil(J, 1, &row, nc, column, value, INSERT_VALUES);
        }
        for (l = ModelAData::Nphi; l < ModelAData::Ndof; l++) {
          // we define the column
          PetscInt nc = 0;
          MatStencil row, column[10];
          PetscScalar value[10];
          // here we insert the position of the row
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = l;
          // here we define de position of the non-vansih column for the given
          // row in total there are 7*4 entries and nc is the total number of
          // column per row x direction
          column[nc].i = i - 1;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hx * hx);
          column[nc].i = i + 1;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hx * hx);
          // y direction
          column[nc].i = i;
          column[nc].j = j - 1;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hy * hy);
          column[nc].i = i;
          column[nc].j = j + 1;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = -1. / (hy * hy);
          // z direction
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k - 1;
          column[nc].c = l;
          value[nc++] = -1. / (hz * hz);
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k + 1;
          column[nc].c = l;
          value[nc++] = -1. / (hz * hz);

          // The central element need a loop over the flavour index of the
          // column (is a full matrix in the flavour index )
          column[nc].i = i;
          column[nc].j = j;
          column[nc].k = k;
          column[nc].c = l;
          value[nc++] = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz);

          // here we set the matrix
          MatSetValuesStencil(J, 1, &row, nc, column, value, INSERT_VALUES);
        }
      }
    }
  }
  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
  return (0);
}



LFHBSplit::LFHBSplit(ModelA &in)
    : lf(in), hbPhi(in), hbN(in), dtHB(0.004)
    {

}


bool LFHBSplit::step(const double &dt) {
  lf.step(dt);
  int nhb = dt / dtHB;
  double tdiff = 0;
  for(int i = 0; i < nhb - 1; ++i){
    hbPhi.step(dtHB);
    hbN.step(dtHB);
    tdiff += dtHB;
  }
  hbPhi.step(dt - tdiff);
  hbN.step(dt - tdiff);

}
