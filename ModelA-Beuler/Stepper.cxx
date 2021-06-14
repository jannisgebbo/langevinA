#include "Stepper.h"
#include "ModelA.h"

//! Helper function for the forward euler step
o4_node localtimederivative(o4_node *phi, o4_node *phixminus, o4_node *phixplus,
                            o4_node *phiyminus, o4_node *phiyplus,
                            o4_node *phizminus, o4_node *phizplus, void *ptr) {

  ModelA *model = (ModelA *)ptr;
  const ModelAData &data = model->data;
  PetscReal hx = data.hX();
  PetscReal hy = data.hY();
  PetscReal hz = data.hZ();
  o4_node phidot;

  // l is an index for our vector. l=0,1,2,3 is phi,l=4,5,6 is V and l=7,8,9 is
  // A std::cout << "bad Job: " ; computing phi squared
  PetscScalar phisquare = 0.;
  for (PetscInt l = 0; l < ModelAData::Ndof; l++) {
    phisquare = phisquare + phi->f[l] * phi->f[l];
  }

  for (PetscInt l = 0; l < ModelAData::Ndof; l++) {
    PetscScalar ucentral = phi->f[l];
    PetscScalar uxx =
        (-2.0 * ucentral + phixminus->f[l] + phixplus->f[l]) / (hx * hx);
    PetscScalar uyy =
        (-2.0 * ucentral + phiyminus->f[l] + phiyplus->f[l]) / (hy * hy);
    PetscScalar uzz =
        (-2.0 * ucentral + phizminus->f[l] + phizplus->f[l]) / (hz * hz);

    phidot.f[l] =
        data.gamma * (uxx + uyy + uzz) -
        data.gamma * (data.mass + data.lambda * phisquare) * ucentral +
        (l == 0 ? data.gamma * data.H : 0.);
  }

  return (phidot);
};

bool ForwardEuler::step(const double &dt) {

  if (withNoise) {
    ModelARndm->fillVec(noise);
  } else {
    VecZeroEntries(noise);
  }

  const ModelAData &data = model->data;
  PetscInt i, j, k, l, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;
  DM da = model->domain;

  // define and allocate a local vector
  Vec localU;
  DMGetLocalVector(da, &localU);

  // take the global vector solution and distribute to the local vector localU
  DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localU);
  DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localU);

  // From the vector define the pointer for the field u and the rhs f
  o4_node ***phi;
  DMDAVecGetArray(da, localU, &phi);

  o4_node ***phinew;
  DMDAVecGetArray(da, model->solution, &phinew);

  o4_node ***gaussiannoise;
  DMDAVecGetArrayRead(da, noise, &gaussiannoise);

  o4_node ***phidot;
  DMDAVecGetArray(da, model->phidot, &phidot);

  // Get The local coordinates
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // This actually compute the right hand side
  // PetscScalar uxx,uyy,uzz,ucentral,phisquare;
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {

        o4_node derivative = localtimederivative(
            &phi[k][j][i], &phi[k][j][i - 1], &phi[k][j][i + 1],
            &phi[k][j - 1][i], &phi[k][j + 1][i], &phi[k - 1][j][i],
            &phi[k + 1][j][i], model);

        for (l = 0; l < ModelAData::Ndof; l++) {
          phidot[k][j][i].f[l] =
              derivative.f[l] +
              PetscSqrtReal(2. * data.gamma / dt) * gaussiannoise[k][j][i].f[l];

          // here you want to put the formula for the euler step F(phi)=0
          phinew[k][j][i].f[l] += dt * phidot[k][j][i].f[l];
        }
      }
    }
  }
  // We need to restore the vector in U and F
  DMDAVecRestoreArray(da, localU, &phi);
  DMDAVecRestoreArray(da, model->solution, &phinew);

  // Destroy the local vector
  DMRestoreLocalVector(da, &localU);

  DMDAVecRestoreArrayRead(da, noise, &gaussiannoise);
  // DMDAVecRestoreArrayRead(da,model->previoussolution,&oldphi);

  DMDAVecRestoreArray(da, model->phidot, &phidot);

  return true;
}

/////////////////////////////////////////////////////////////////////////

BackwardEuler::BackwardEuler(ModelA &in, bool wnoise) : model(&in), withNoise(wnoise) {

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

  // define a local vector
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
  o4_node ***phi, ***f;
  DMDAVecGetArrayRead(da, localU, &phi);
  DMDAVecGetArray(da, F, &f);

  o4_node ***gaussiannoise;
  DMDAVecGetArrayRead(da, stepper->noise, &gaussiannoise);

  o4_node ***oldphi;
  DMDAVecGetArrayRead(da, model->previoussolution, &oldphi);

  o4_node ***phidot;
  DMDAVecGetArray(da, model->phidot, &phidot);

  // Get the local coordinates
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // This actually computes the right hand side
  PetscScalar uxx, uyy, uzz, ucentral, phisquare;
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {
        phisquare = 0.;
        for (l = 0; l < data.Ndof; l++) {
          phisquare = phisquare + phi[k][j][i].f[l] * phi[k][j][i].f[l];
        }
        for (l = 0; l < data.Ndof; l++) {
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

          phidot[k][j][i].f[l] =
              data.gamma * (uxx + uyy + uzz) -
              data.gamma * (data.mass + data.lambda * phisquare) * ucentral +
              (l == 0 ? data.gamma * data.H : 0.) +
              PetscSqrtReal(2. * data.gamma / stepper->deltat) *
                  gaussiannoise[k][j][i].f[l];

          // here you want to put the formula for the euler step F(phi)=0
          f[k][j][i].f[l] = -phi[k][j][i].f[l] + oldphi[k][j][i].f[l] +
                            stepper->deltat * phidot[k][j][i].f[l];
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

  DMDAVecRestoreArray(da, model->phidot, &phidot);

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
  o4_node ***phi;
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
        for (l = 0; l < ModelAData::Ndof; l++) {
          phisquare = phisquare + phi[k][j][i].f[l] * phi[k][j][i].f[l];
        }
        for (l = 0; l < ModelAData::Ndof; l++) {
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
          for (PetscInt m = 0; m < ModelAData::Ndof; m++) {
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

SemiImplicitBEuler::SemiImplicitBEuler(ModelA &in, bool wnoise) : model(&in), withNoise(wnoise) {
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

  o4_node ***bI;
  DMDAVecGetArray(model->domain, rhs, &bI);

  o4_node ***phiI;
  DMDAVecGetArray(model->domain, model->solution, &phiI);

  o4_node ***xiI;
  DMDAVecGetArray(model->domain, noise, &xiI);

  PetscInt i, j, k, l, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;

  DMDAGetCorners(model->domain, &xstart, &ystart, &zstart, &xdimension,
                 &ydimension, &zdimension);

  const ModelAData &data = model->data;
  double dtg = dt * data.gamma;
  double m2 = data.mass;
  double lambda = data.lambda;
  double H[ModelAData::Ndof] = {data.H, 0, 0, 0} ;

  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {

        o4_node &phi = phiI[k][j][i];
        o4_node &b = bI[k][j][i];
        o4_node &xi = xiI[k][j][i];

        double phi2 = 0.;
        for (l = 0; l < ModelAData::Ndof; l++) {
          phi2 += phi.f[l] * phi.f[l];
        }

        for (l = 0; l < ModelAData::Ndof; l++) {
          b.f[l] = phi.f[l] / dtg - (m2 * phi.f[l] + lambda * phi2 * phi.f[l]) 
             + H[l] + xi.f[l] * sqrt(2. / dtg);
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
        for (l = 0; l < ModelAData::Ndof; l++) {
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
