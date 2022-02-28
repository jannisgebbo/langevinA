#ifndef MODELASTRUCT
#define MODELASTRUCT

#include "NoiseGenerator.h"
#include "json/json.h"
#include <fstream>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscts.h>
#ifndef MODELA_NO_HDF5
#include <petscviewerhdf5.h>
#endif
#include "make_unique.h"

// POD structure for recording the time stepping
struct ModelATime {
  PetscReal finaltime = 10.;
  PetscReal initialtime = 0.;
  PetscReal deltat = 0.24;
  PetscReal time = 0.;

  PetscReal t() const { return time; }
  PetscReal dt() const { return deltat; }
  PetscReal tinitial() const { return initialtime; }
  PetscReal tfinal() const { return finaltime; }
  void operator+=(const double &dtin) { time += dtin; }
  void reset() { time = initialtime; }

  // The finaltime is be adjusted so that the total number of time steps is an
  // integral number of saveFrequency*dt() units.  The new finaltime is
  // returned.
  PetscReal adjust_finaltime(const int &saveFrequency) {
    finaltime = initialtime + deltat * saveFrequency *
                                  static_cast<int>((finaltime - initialtime) /
                                                   (deltat * saveFrequency));
    return finaltime;
  }

  void read(const Json::Value &params) {
    finaltime = params.get("finaltime", finaltime).asDouble();
    initialtime = params.get("initialtime", initialtime).asDouble();
    deltat = params.get("deltat", deltat).asDouble();
    time = initialtime;
  }

  void print() {
    // Time Stepping
    PetscPrintf(PETSC_COMM_WORLD, "finaltime = %e\n", finaltime);
    PetscPrintf(PETSC_COMM_WORLD, "initialtime = %e\n", initialtime);
    PetscPrintf(PETSC_COMM_WORLD, "deltat  = %e\n", deltat);
  }
};

// POD structure for the thermodynamic and transport coefficients.
struct ModelACoefficients {

  PetscReal mass0 = -4.70052;
  PetscReal dmassdt = 0.;

  PetscReal lambda = 4.;
  PetscReal H = 0.003;
  PetscReal chi = 5;

  PetscReal gamma = 1.;
  PetscReal diffusion = 0.3333333;

  // Returns the value of the mass at a given time t.
  PetscReal mass(const double &t) const { return mass0 + dmassdt * t; }

  PetscReal sigma() const { return diffusion * chi; }
  PetscReal D() const { return diffusion; }

  void read(const Json::Value &params) {
    mass0 = params.get("mass0", mass0).asDouble();
    dmassdt = params.get("dmassdt", 0.).asDouble();

    lambda = params.get("lambda", lambda).asDouble();
    H = params.get("H", H).asDouble();
    chi = params.get("chi", chi).asDouble();

    gamma = params.get("gamma", gamma).asDouble();
    diffusion = params.get("diffusion", 1. / 3. * gamma).asDouble();
  }

  void print() {
    PetscPrintf(PETSC_COMM_WORLD, "mass0 = %e\n", mass0);
    PetscPrintf(PETSC_COMM_WORLD, "dmassdt = %e\n", dmassdt);

    PetscPrintf(PETSC_COMM_WORLD, "lambda = %e\n", lambda);
    PetscPrintf(PETSC_COMM_WORLD, "H = %e\n", H);
    PetscPrintf(PETSC_COMM_WORLD, "chi = %e\n", chi);
    PetscPrintf(PETSC_COMM_WORLD, "gamma = %e\n", gamma);
    PetscPrintf(PETSC_COMM_WORLD, "diffusion = %e\n", diffusion);
  }
};

struct ModelAHandlerData {

  std::string evolverType = "PV2HBSplit23";

  // random seed
  PetscInt seed = 10;

  // If we are to restore
  bool restart = false;

  // Options controlling the output. The outputfiletag labells the run. All
  // output files are tag_foo.txt, or tag_bar.h5
  std::string outputfiletag = "o4output";
  PetscInt saveFrequency = 3;
  PetscInt writeFrequency = -1;

  bool eventmode = false;
  int nevents = 1;
  int last_stored_event = -1;
  int current_event = 0;
  double thermalization_time = 0. ;

  void read(Json::Value &params) {
    evolverType = params.get("evolverType", evolverType).asString();
    seed = (PetscInt)params.get("seed", seed).asInt();
    restart = params.get("restart", false).asBool();
    outputfiletag = params.get("outputfiletag", "o4output").asString();
    saveFrequency = params.get("saveFrequency", saveFrequency).asInt();
    writeFrequency = params.get("writeFrequency", writeFrequency).asInt();
    thermalization_time = params.get("thermalization_time", thermalization_time).asDouble() ;

    eventmode = params.get("eventmode", eventmode).asBool();
    nevents = params.get("nevents", nevents).asInt();
    last_stored_event = params.get("last_stored_event", -1).asInt();
    current_event = last_stored_event + 1; 
//
    // This is for restart mode
    params["last_stored_event"] = last_stored_event + nevents;

  }

  void print() {
    PetscPrintf(PETSC_COMM_WORLD, "evolverType = %s\n", evolverType.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "seed = %d\n", seed);
    PetscPrintf(PETSC_COMM_WORLD, "restart = %s\n",
                (restart ? "true" : "false"));
    PetscPrintf(PETSC_COMM_WORLD, "outputfiletag = %s\n",
                outputfiletag.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "saveFrequency = %d\n", saveFrequency);
    PetscPrintf(PETSC_COMM_WORLD, "thermalization_time = %e\n",
                thermalization_time);

    PetscPrintf(PETSC_COMM_WORLD, "eventmode = %s\n",
                (eventmode ? "true" : "false"));
    PetscPrintf(PETSC_COMM_WORLD, "nevents = %d\n", nevents);
    PetscPrintf(PETSC_COMM_WORLD, "last_stored_event = %d\n",
                last_stored_event);
  }
};

class ModelAData {

public:
  // Lattice dimension
  PetscInt NX = 16;
  PetscInt NY = 16;
  PetscInt NZ = 16;

  // Lattice size
  PetscReal LX = 16.;
  PetscReal LY = 16.;
  PetscReal LZ = 16.;

  // Lattice spacing in physical units
  PetscReal hX() const { return LX / NX; }
  PetscReal hY() const { return LY / NY; }
  PetscReal hZ() const { return LZ / NZ; }

  // Number of fields
  static const PetscInt Nphi = 4;
  static const PetscInt NA = 3;
  static const PetscInt NV = 3;
  static const PetscInt Ndof = Nphi + NA + NV;

  // Options controlling the clock
  ModelATime atime;

  // Thermodynamic and transport coefficients
  ModelACoefficients acoefficients;
  // Convenience function for returnin the mass at the current time
  PetscReal mass() const { return acoefficients.mass(atime.t()); }

  // Options for management ;
  ModelAHandlerData ahandler;

public:
  ModelAData(Json::Value &params) {
    // Lattice. By default, NY=NX and NZ=NX.
    NX = params["NX"].asInt();
    NY = NX;
    NZ = NX;

    // By default, dx=dy=dz=1, namely LX=NX, LY=NY, LZ=NZ.
    LX = params.get("LX", NX).asInt();
    LY = LX;
    LZ = LX;

    // Time Stepping
    atime.read(params);
    acoefficients.read(params);
    ahandler.read(params);

    // Micro adjust the final time
    double finaltime = atime.adjust_finaltime(ahandler.saveFrequency);
    params["finaltime"] = finaltime;

    // Printout
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      print();
    }
  }

  //! Print out ModelAData for subsequent reading
  void print() {
    // Lattice
    PetscPrintf(PETSC_COMM_WORLD, "NX = %d\n", NX);
    PetscPrintf(PETSC_COMM_WORLD, "NY = %d\n", NY);
    PetscPrintf(PETSC_COMM_WORLD, "NZ = %d\n", NZ);
    PetscPrintf(PETSC_COMM_WORLD, "LX = %e\n", LX);
    PetscPrintf(PETSC_COMM_WORLD, "LY = %e\n", LY);
    PetscPrintf(PETSC_COMM_WORLD, "LZ = %e\n", LZ);

    // Time Stepping
    atime.print();

    // Transport and thermodynamic coefficients
    acoefficients.print();

    // Mangement parameters
    ahandler.print();
  }
};

/////////////////////////////////////////////////////////////////////////

typedef struct {
  PetscScalar f[ModelAData::Nphi];
  PetscScalar A[ModelAData::NA];
  PetscScalar V[ModelAData::NV];
} G_node;

typedef struct {
  PetscScalar x[ModelAData::Ndof];
} data_node;

class ModelA {

public:
  // Plain old data describing model A
  ModelAData data;

  // Domain descriptor
  DM domain;

  // Solution
  Vec solution;

  // Previous solution
  Vec previoussolution;

  // Rank of this processor
  int rank;

  //! Construct the grid and initialize the fields according to
  //! the configuration parameters in the input ModelAData structure
  ModelA(const ModelAData &in) : data(in) {

    PetscInt stencil_width = 1;
    PetscInt Ndof = ModelAData::Nphi + ModelAData::NA + ModelAData::NV;
    DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                 DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, data.NX, data.NY,
                 data.NZ, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, Ndof,
                 stencil_width, NULL, NULL, NULL, &domain);

    DMSetFromOptions(domain);
    DMSetUp(domain);

    DMCreateGlobalVector(domain, &solution);
    VecDuplicate(solution, &previoussolution);

    // Setup the random number generation
    const auto &ahandler = data.ahandler;
    ModelARndm = make_unique<NoiseGenerator>(ahandler.seed);
    if (ahandler.restart) {
      ModelARndm->read(ahandler.outputfiletag);
    }

    // Printout store the rank
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }

  void finalize() {
    const auto &ahandler = data.ahandler;
    ModelARndm->write(ahandler.outputfiletag);
    write(ahandler.outputfiletag);
    VecDestroy(&previoussolution);
    VecDestroy(&solution);
    DMDestroy(&domain);
  }

  //! Reads in a stored initial condition from Derek's file.
  //! This is a helper function for initialize
  PetscErrorCode read(const std::string fnamein) {
    std::string fname = fnamein + "_save.h5";
#ifndef MODELA_NO_HDF5
    PetscViewer initViewer;
    PetscBool flg;
    PetscErrorCode ierr = PetscTestFile(fname.c_str(), '\0', &flg);
    if (!flg) {
      throw(
          std::string("Unable to open file in restart mode with filename = ") +
          fname);
    }
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fname.c_str(), FILE_MODE_READ,
                               &initViewer);
    CHKERRQ(ierr);
    PetscViewerSetFromOptions(initViewer);
    PetscObjectSetName((PetscObject)solution, "o4fields");
    ierr = VecLoad(solution, initViewer);
    CHKERRQ(ierr);
    PetscViewerDestroy(&initViewer);
    return ierr;
#else
    throw("ModelA::read Unable to load from file without HDF5 support. \n");
    return 0;
#endif
  }

  //! Reads in a stored initial condition from Derek's file.
  //! This is a helper function for initialize
  PetscErrorCode write(const std::string fnamein) {
    std::string fname = fnamein + "_save.h5";
#ifndef MODELA_NO_HDF5
    PetscViewer initViewer;
    PetscErrorCode ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fname.c_str(),
                                              FILE_MODE_WRITE, &initViewer);
    CHKERRQ(ierr);
    PetscViewerSetFromOptions(initViewer);
    PetscObjectSetName((PetscObject)solution, "o4fields");
    ierr = VecView(solution, initViewer);
    CHKERRQ(ierr);
    PetscViewerDestroy(&initViewer);
    return 0;
#else
    PetscPrintf(
        "ModelA::write Unable to write to a file wihtout HDF5 support\n");
    return 0;
#endif
  }

  // Initialize the vector solution. If coldStart is false  then read in the
  // data from Derek's file. Otherwise fill with  random numbers, or if
  // zeroStart is true set to zero. Finally  if a function is provided,
  // f(x,y,z,L, params), this function will be used.
  PetscErrorCode initialize(double (*func)(const double &x, const double &y,
                                           const double &z, const int &L,
                                           void *params) = 0,
                            void *params = 0) {

    const auto &ahandler = data.ahandler;
    if (ahandler.restart) {
      try {
        read(ahandler.outputfiletag);
        return (0);
      } catch (const std::string &error) {
        std::cout << error << std::endl;
        std::cout << "Continuing with zerostart mode." << std::endl;
      }
    }

    // Compute the lattice spacing
    PetscReal hx = data.hX();
    PetscReal hy = data.hY();
    PetscReal hz = data.hZ();

    // This Get a pointer to do the calculation
    PetscScalar ****u;
    DMDAVecGetArrayDOF(domain, solution, &u);

    // Get the Local Corner od the vector
    PetscInt i, j, k, L, xstart, ystart, zstart, xdimension, ydimension,
        zdimension;

    DMDAGetCorners(domain, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                   &zdimension);

    // This is the actual computation of the thing
    for (k = zstart; k < zstart + zdimension; k++) {
      PetscReal z = k * hz;
      for (j = ystart; j < ystart + ydimension; j++) {
        PetscReal y = j * hy;
        for (i = xstart; i < xstart + xdimension; i++) {
          PetscReal x = i * hx;
          for (L = 0; L < ModelAData::Ndof; L++) {
            if (func) {
              u[k][j][i][L] = func(x, y, z, L, params);
            } else {
              u[k][j][i][L] = 0.;
            }
          }
        }
      }
    }
    DMDAVecRestoreArrayDOF(domain, solution, &u);
    return (0);
  }
};

#endif
