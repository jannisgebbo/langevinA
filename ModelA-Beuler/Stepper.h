#ifndef STEPPER_H
#define STEPPER_H

#include "ModelA.h"

class Stepper {
public:
  virtual bool step(const double &dt) = 0;
  virtual void finalize() = 0;
  virtual ~Stepper() = default;
};

class ForwardEuler : public Stepper {
public:
  ForwardEuler(ModelA &in, bool wnoise = true);
  bool step(const double &dt);
  void finalize();
  ~ForwardEuler() { ; }

private:
  ModelA *model;
  bool withNoise;
  Vec noise;
};

/////////////////////////////////////////////////////////////////////////

class BackwardEuler : public Stepper {
public:
  BackwardEuler(ModelA &in, bool wnoise = true);
  bool step(const double &dt);
  void finalize();
  ~BackwardEuler() { ; }

private:
  ModelA *model;
  bool withNoise;
  SNES Solver; // The solver that does the work

  Mat jacobian;
  Vec noise;
  Vec auxsolution;

  // The time step
  double deltat;

  // Evaluates the RHS function
  static PetscErrorCode FormFunction(SNES snes, Vec U, Vec F, void *ptr);
  // Evaluates the Jacobian function. This is just a stub with the petsc
  // interface which calls the version below
  static PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre,
                                     void *ptr);
  // Evaluates the Jacobian function
  static PetscErrorCode FormJacobianGeneric(SNES snes, Vec U, Mat J, Mat Jpre,
                                            const double &dt, ModelA *model);
};

/////////////////////////////////////////////////////////////////////////
// We solve:
//
// phidot =  gamma (L phi - partial V/partial phi + H) + xi
//
// where L is the laplacian.  Here xi has variance 2 gamma/dt.  The semi
// implicit discretization discretizes this as
//
// (1 - dtg L) phi^{n+1} = phi^n  + dtg (-partial V/partial phi + H)
//                                              + xin*sqrt(2 dtg)
//
// We have icorporated gamma into the definition of dt, i.e.  dtg = dt*gamma,
// and denote xin as noise with unit variance.  We slightly rearrange the
// equation, by dividing by dtg yield the form used in the code
//
//   ( 1/dtg - L ) phi^n+1 = phi^n/dtg + (-partial V/partial phi + H)
//                                       +   xin * sqrt(2/dtg)
//
// Let J = -L, and we note that this is positive definite.
//
// We will create the the matrix  A=(1/dtg + J)  by using MatCopy, which copies
// the matrix J to A, and then MatScale  which implements  the shift A-> A +
// a*I  where I is the indenitity matrix. In this case a=1/dtg.
//
// On a technical note the matrix A can be created using MatConvert.
class SemiImplicitBEuler : public Stepper {
public:
  SemiImplicitBEuler(ModelA &in, bool wnoise = true);
  bool step(const double &dt);
  void finalize();
  ~SemiImplicitBEuler() { ; }
  static PetscErrorCode Form3PointLaplacian(DM da, Mat J, const double &hx,
                                            const double &hy, const double &hz);

private:
  ModelA *model;
  bool withNoise;

  Vec rhs;
  Vec noise;
  Mat J;
  Mat A;

  KSP ksp;
};

#endif
