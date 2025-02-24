#ifndef MEASURER_OUTPUT
#define MEASURER_OUTPUT

#include "measurer.h"
#include <array>
#include <complex>
#include <vector>

#ifndef MODELA_NO_HDF5
#include "ntuple.h"
#else
#define MODELA_TXTOUTPUT
#endif

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
#ifndef MODELA_NO_HDF5
class measurer_output_fasthdf5 : public measurer_output {
public:
  measurer_output_fasthdf5(Measurer *in, const std::string filename,
                           const PetscFileMode &mode);
  ~measurer_output_fasthdf5();
  virtual void save(const std::string &what = "") override;

private:
  Measurer *measure;
  hid_t file_id;

  // One dimensional quantities
  std::unique_ptr<ntuple<1>> scalars;
  // Time and mass of the measurement is also recorded.
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
#endif
