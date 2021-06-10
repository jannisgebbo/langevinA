#ifndef STEPPER_H
#define STEPPER_H

#include "ModelA.h"

class Stepper {
   public:
      virtual bool step(const double &dt) = 0;
      virtual void finalize() = 0 ;
      virtual ~Stepper() = default;
};

class ForwardEuler : public Stepper {
   public:
      ForwardEuler(ModelA &in) : model(&in) {;}
      bool step(const double &dt);
      void finalize() {;}
      ~ForwardEuler() {;}

   private:
      ModelA *model;
};

class BackwardEuler : public Stepper {
   public:
      BackwardEuler(ModelA &in) ;
      bool step(const double &dt);
      void finalize() ;
      ~BackwardEuler() {;}

   private:

     ModelA *model;
     SNES Solver; // The solver that does the workj

     // The time step
     double deltat ;

     // Evaluates the RHS function
     static PetscErrorCode FormFunction(SNES snes, Vec U, Vec F, void *ptr) ;
     // Evaluates the Jacobian function
     static PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre, void *ptr) ;

};


#endif

