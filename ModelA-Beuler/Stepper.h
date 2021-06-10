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
  ForwardEuler(ModelA &in) : model(&in) {
    VecDuplicate(model->solution, &noise);
  }
  bool step(const double &dt);
  void finalize() { VecDestroy(&noise); }
  ~ForwardEuler() { ; }

private:
  ModelA *model;
  Vec noise;
};

class BackwardEuler : public Stepper {
public:
  BackwardEuler(ModelA &in);
  bool step(const double &dt);
  void finalize();
  ~BackwardEuler() { ; }

private:
  ModelA *model;
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

#endif
