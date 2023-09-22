#ifndef MEASURER
#define MEASURER

#include "ModelA.h"
#include "make_unique.h"
#include "nvector.h"
#include <array>
#include <complex>
#include <fftw3.h>
#include <vector>

#ifndef MODELA_NO_HDF5
#include "ntuple.h"
#else
#define MODELA_TXTOUTPUT
#endif

class Measurer;

// Interface for output of measurements
class measurer_output {
public:
  virtual ~measurer_output() { ; }
  virtual void save(const std::string &what) = 0;
};

////////////////////////////////////////////////////////////////////////

// Minimal output of scalar array to a text file
class measurer_output_txt : public measurer_output {
public:
  measurer_output_txt(Measurer *in);
  ~measurer_output_txt();
  virtual void save(const std::string &what);

private:
  Measurer *measure;
  PetscViewer averages_asciiviewer;
};

////////////////////////////////////////////////////////////////////////
class measurer_fft {

private:
  // Create the storage space for the plan
  std::vector<double> in_ptr;
  std::vector<std::complex<double>> out_ptr;
  fftw_plan plan;

public:
  measurer_fft(const size_t N) : in_ptr(N, 0.), out_ptr(N / 2 + 1, 0.) {
    // Create the plane for the fourier transform
    plan = fftw_plan_dft_r2c_1d(
        N, in_ptr.data(), reinterpret_cast<fftw_complex *>(out_ptr.data()),
        FFTW_MEASURE);
  }
  ~measurer_fft() { fftw_destroy_plan(plan); }

  void execute_row(double *in, std::complex<double> *out) {
    for (size_t i = 0; i < in_ptr.size(); i++) {
      in_ptr[i] = in[i];
    }
    fftw_execute(plan);
    for (size_t i = 0; i < out_ptr.size(); i++) {
      out[i] = out_ptr[i] / in_ptr.size();
    }
  }

  // Compute the fourier transform of an array in
  void execute(nvector<double, 2> &in, nvector<std::complex<double>, 2> &out) {
    for (int i = 0; i < in.N[0]; i++) {
      execute_row(&in(i, 0), &out(i, 0));
    }
  }
};

////////////////////////////////////////////////////////////////////////
#ifndef MODELA_NO_HDF5
class measurer_output_fasthdf5 : public measurer_output {
public:
  measurer_output_fasthdf5(Measurer *in, const std::string filename,
                           const PetscFileMode &mode);
  ~measurer_output_fasthdf5();
  virtual void save(const std::string &what) override;

private:
  Measurer *measure;
  hid_t file_id;

  // One dimensional quantities
  std::unique_ptr<ntuple<1>> scalars;
  std::unique_ptr<ntuple<1>> timeout;

  // Output
  std::unique_ptr<ntuple<2>> wallx;
  std::unique_ptr<ntuple<2>> wally;
  std::unique_ptr<ntuple<2>> wallz;

  // Output of fourier info
  std::unique_ptr<ntuple<3>> wallx_k;
  std::unique_ptr<ntuple<3>> wally_k;
  std::unique_ptr<ntuple<3>> wallz_k;

  // Output of fourier info rotated
  std::unique_ptr<ntuple<3>> wallx_k_rotated;
  std::unique_ptr<ntuple<3>> wally_k_rotated;
  std::unique_ptr<ntuple<3>> wallz_k_rotated;

  // Output of phase info
  std::unique_ptr<ntuple<2>> wallx_phase;
  std::unique_ptr<ntuple<2>> wally_phase;
  std::unique_ptr<ntuple<2>> wallz_phase;

  // Output of fourier phase info
  std::unique_ptr<ntuple<3>> wallx_phase_k;
  std::unique_ptr<ntuple<3>> wally_phase_k;
  std::unique_ptr<ntuple<3>> wallz_phase_k;
};
#endif
////////////////////////////////////////////////////////////////////////

// class Measurment {
//    void initialize(Vec *solution, const double &time) =0 ;
//    void measure(Vec *solution, const double &time) =0 ;
//    void finalize(Vec *solution, const double &time) =0 ;
// } ;
//  std::map<std::string,Measurement>  measurements ;

class Measurer {

public:
  Measurer(ModelA *ptr, const std::string &filename,
           const PetscFileMode &mode = FILE_MODE_WRITE)
      : model(ptr) {
    N = model->data.NX;
    if (model->data.NX != model->data.NY || model->data.NX != model->data.NZ) {
      PetscPrintf(
          PETSC_COMM_WORLD,
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
      throw(
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }

    // Create the hdf5 view
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      measurer_out =
          std::make_unique<measurer_output_fasthdf5>(this, filename, mode);
    }
    fftw = make_unique<measurer_fft>(N);

    wallX.resize(NObs, N);
    wallY.resize(NObs, N);
    wallZ.resize(NObs, N);

    wallX_k.resize(NObs, N / 2 + 1);
    wallY_k.resize(NObs, N / 2 + 1);
    wallZ_k.resize(NObs, N / 2 + 1);

    wallX_k_rotated.resize(NObsRotated, N / 2 + 1);
    wallY_k_rotated.resize(NObsRotated, N / 2 + 1);
    wallZ_k_rotated.resize(NObsRotated, N / 2 + 1);

    wallXPhase.resize(NObsPhase, N);
    wallYPhase.resize(NObsPhase, N);
    wallZPhase.resize(NObsPhase, N);

    wallXPhase_k.resize(NObsPhase, N / 2 + 1);
    wallYPhase_k.resize(NObsPhase, N / 2 + 1);
    wallZPhase_k.resize(NObsPhase, N / 2 + 1);
  }

  virtual ~Measurer() {}

  void finalize() {}

  // Periodically measure the solution writing
  void measure(Vec *solution) {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    computeSliceAverage(solution);
    computeSliceAveragePhase(solution);

    if (rank == 0) {
      computeDerivedObs();
      measurer_out->save("phi");
    }
  }
  const std::vector<PetscScalar> &getOAverage() const { return OAverage; }

private:
  void computeSliceAverage(Vec *solution);
  void computeSliceAveragePhase(Vec *solution);
  void computeDerivedObs();

  ModelA *model;
  PetscInt N;

  // Arrays of size Nobs contain X=(phi[1..Nphi], q[1...Nq], phi2)
  static const PetscInt NObs =
      ModelAData::Nphi + ModelAData::NA + ModelAData::NV + 1;

  // Scalar data
  static const PetscInt NScalars = NObs;
  std::vector<PetscScalar> OAverage;

  // First dimension is NObs, last is spatial index x=0...N
  nvector<PetscScalar, 2> wallX;
  nvector<PetscScalar, 2> wallY;
  nvector<PetscScalar, 2> wallZ;

  // First dimension is NObs, last dimension is fourier index k=0..N/2+1
  nvector<std::complex<double>, 2> wallX_k;
  nvector<std::complex<double>, 2> wallY_k;
  nvector<std::complex<double>, 2> wallZ_k;

  // First dimension is NObsRotated, last dimension is fourier index.
  // Array of size NobsRotate contains XPhase = (sigma, pi[1..Nphi],
  // q[1..2*Nphi], phi2)
  static const PetscInt NObsRotated = 3 * ModelAData::Nphi + 2;
  nvector<std::complex<double>, 2> wallX_k_rotated;
  nvector<std::complex<double>, 2> wallY_k_rotated;
  nvector<std::complex<double>, 2> wallZ_k_rotated;

  // Array of size NobsPhase contains XPhase = (sigma, pi[1..Nphi],
  // q[1..2*Nphi], phi2)
  // The first dimension is NObsPhase, the second domenions is the spatial index
  static const PetscInt NObsPhase = 3 * ModelAData::Nphi + 2;
  nvector<PetscScalar, 2> wallXPhase;
  nvector<PetscScalar, 2> wallYPhase;
  nvector<PetscScalar, 2> wallZPhase;

  // First dimension is NObsPhase, last dimension is the fourier index 0..N/2+1
  nvector<std::complex<double>, 2> wallXPhase_k;
  nvector<std::complex<double>, 2> wallYPhase_k;
  nvector<std::complex<double>, 2> wallZPhase_k;

  // FFT engine using the fftw3 library
  std::unique_ptr<measurer_fft> fftw;

  friend class measurer_output_fasthdf5;
  std::unique_ptr<measurer_output> measurer_out;

  friend class measurer_output_txt;
};

#endif
