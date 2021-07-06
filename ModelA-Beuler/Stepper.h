#ifndef STEPPER_H
#define STEPPER_H

#include "ModelA.h"

class Stepper {
public:
  virtual bool step(const double &dt) = 0;
  virtual void finalize() = 0;
  virtual ~Stepper() = default;
};

class IdealLFStepper : public Stepper {
public:
  IdealLFStepper(ModelA &in);

  bool step(const double &dt) override;
  virtual bool diffusive_step(const double &dt) = 0;
  virtual bool ideal_step(const double &dt);
  // virtual void finalize() = 0;
  virtual ~IdealLFStepper() {}

protected:
  ModelA *model;
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

class ForwardEulerSplit : public IdealLFStepper {
public:
  ForwardEulerSplit(ModelA &in, bool wnoise = true);
  void finalize() override;
  bool diffusive_step(const double &dt) override;
  ~ForwardEulerSplit() { ; }

private:
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

//! Helper class to monitor the jumps in phi at each time step, and to
//! record the number of successful steps
class o4_stepper_monitor {
public:
  long int down_counts; //!< number of steps downwards
  long int up_counts;   //!< number of upward trial steps
  long int up_yes;      //!< number of accepted upward steps
  long int up_no;       //!< number of rejected upward steps
  double down_dS;       //!< The change in action summed over downward steps
  double up_dS;         //!< The change in action summed over upward steps

public:
  o4_stepper_monitor() {
    down_counts = 0;
    up_counts = 0;
    up_yes = 0;
    up_no = 0;
    down_dS = 0;
    up_dS = 0;
  }
  void increment_up_yes(const double &deltaS) {
    up_counts++;
    up_yes++;
    up_dS += deltaS;
  }
  void increment_up_no(const double &deltaS) {
    up_counts++;
    up_no++;
    up_dS += 0.;
  }
  void increment_down(const double &deltaS) {
    down_counts++;
    down_dS += deltaS;
  }

  void print(FILE *fh) {

    double tot = down_counts + up_counts;
    double prob = up_yes / tot;
    double mean_dS = up_dS / up_yes;
    fprintf(fh, "==== Statistics of Heat Bath Type Steps ====\n");
    fprintf(fh,
            "dS >0: accepts = %ld; probability of up accept = %f; mean dS=%f\n",
            up_yes, prob, mean_dS);

    prob = up_no / tot;
    fprintf(fh,
            "dS =0: rejects = %ld; probability of up reject = %f; mean dS=%f\n",
            up_no, prob, 0.);

    prob = down_counts / tot;
    mean_dS = down_dS / down_counts;
    fprintf(fh,
            "dS <0: counts  = %ld; probability of down steps= %f; mean dS=%f\n",
            down_counts, prob, mean_dS);

    prob = up_counts / tot;
    mean_dS = up_dS / up_counts;
    fprintf(fh,
            "dS>=0: counts  = %ld; probability of non-neg.  = %f; mean dS=%f\n",
            up_counts, prob, mean_dS);

    prob = up_yes / static_cast<double>(up_counts);
    mean_dS = up_dS / up_counts;
    fprintf(fh,
            "dS>=0: counts  = %ld; probability of accepting when up  = %f; "
            "mean dS=%f\n",
            up_yes, prob, mean_dS);
    prob = up_no / static_cast<double>(up_counts);
    fprintf(fh,
            "dS>=0: counts  = %ld; probability of rejecting when up  = %f; "
            "mean dS=%f\n",
            up_no, prob, mean_dS);
    fprintf(fh, "============================================\n");
  }
};

// The heat bath step for the phi fields
class EulerLangevinHB : public Stepper {
public:
  EulerLangevinHB(ModelA &in);
  bool step(const double &dt);
  void finalize();
  ~EulerLangevinHB() { ; }

private:
  ModelA *model;
  o4_stepper_monitor monitor;

  Vec phi_local;
};

// The heat bath step for the charge fields
class ModelGChargeHB : public Stepper {
public:
  ModelGChargeHB(ModelA &in);
  bool step(const double &dt);
  void finalize();
  ~ModelGChargeHB() { ; }

private:
  ModelA *model;
  o4_stepper_monitor qmonitor;

  Vec phi_local;
  Vec dn_local;
};

// These are helper structures for ModelGChargeHB
struct g_face_case {
  int eoA; // zero or one / even or odd of site A
  int iB;  // either zero or one (indexing of the B cell relative to A)
  int jB;  // either zero or one  (indexing of the B cell relative to A)
  int kB;  // either zero or one  (indexing of the B cell relative to A)
};

extern g_face_case g_face_cases[3][2];

PetscScalar modelg_update_charge_pair(const double &chi, const double &rms,
                                      const PetscScalar &nA,
                                      const PetscScalar &nB,
                                      o4_stepper_monitor &monitor);

/////////////////////////////////////////////////////////////////////////
// The Crank Nicholson step for q fields
class ModelGChargeCN : public Stepper {
public:
  ModelGChargeCN(ModelA &in, bool withNoise = true);
  bool step(const double &dt);
  void finalize();
  ~ModelGChargeCN() { ; }
  static PetscErrorCode Form3PointLaplacian(DM da, Mat J, const double &hx,
                                            const double &hy, const double &hz);

private:
  ModelA *model;
  bool withNoise;

  Vec rhs;
  Vec noise_local;
  Vec dn;
  Mat J;
  Mat A;

  KSP ksp;
};
#endif
