#ifndef MODELASTRUCT
#define MODELASTRUCT

#include "NoiseGenerator.h"
#include "parameterparser/parameterparser.h"
#include <fstream>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscts.h>
#include <petscviewerhdf5.h>

struct ModelAData {
  // Lattice dimension
  PetscInt NX = 16;
  PetscInt NY = 16;
  PetscInt NZ = 16;

  // Lattice size
  PetscReal LX = 1.;
  PetscReal LY = 1.;
  PetscReal LZ = 1.;

  // Lattice spacing in physical units
  PetscReal hX() const { return LX / NX; }
  PetscReal hY() const { return LY / NY; }
  PetscReal hZ() const { return LZ / NZ; }

  // Number of fields
  static const PetscInt Ndof = 4;

  // Options controlling the time stepping:
  PetscReal finaltime = 20.;
  PetscReal initialtime = 0.;
  PetscReal deltat = 0.01;
  // 1 is BEuler, 2 is FEuler
  PetscInt evolverType = 2;

  // The parameters for the action.
  // The mass in the action has 1/2 and is actually the mass squared and the
  // quartic has intead 1/4.
  PetscReal mass = -5.;
  PetscReal lambda = 5.;
  PetscReal gamma = 1.;
  PetscReal H = 0.;

  // random seed
  PetscInt seed = 10;

  // Options controlling the initial condition
  std::string initFile = "x";
  bool zeroStart;
  bool coldStart;

  // Options controlling the output. The outputfiletag
  // labells the run. All output files are tag_foo.txt, or tag_bar.h5
  std::string outputfiletag = "o4output";
  PetscReal saveFrequencyInTime;
  PetscInt saveFrequency;
  bool verboseMeasurements;

  ModelAData(FCN::ParameterParser &params) {
    // Lattice. By default, NY=NX and NZ=NX.
    NX = params.get<int>("NX");
    NY = params.get<int>("NY", NX);
    NZ = params.get<int>("NZ", NX);

    // By default, dx=dy=dz=1, namely LX=NX, LY=NY, LZ=NZ.

    LX = params.get<double>("LX", NX);
    LY = params.get<double>("LY", NY);
    LZ = params.get<double>("LZ", NZ);

    // Time Stepping
    finaltime = params.get<double>("finaltime");
    initialtime = params.get<double>("initialtime");
    deltat = params.get<double>("deltat");
    evolverType = params.get<int>("evolverType");

    // Action
    mass = params.get<double>("mass");
    lambda = params.get<double>("lambda");
    gamma = params.get<double>("gamma");
    H = params.get<double>("H");

    seed = (PetscInt)params.getSeed("seed");

    // Control initialization
    initFile = params.get<std::string>("initFile", "x");
    if (initFile == "x") {
      coldStart = true;
    } else {
      coldStart = false;
    }
    zeroStart = params.get<bool>("zeroStart", false);

    // Control outputs
    outputfiletag = params.get<std::string>("outputfiletag", "o4output");
    saveFrequencyInTime = params.get<double>("saveFrequencyInTime");
    verboseMeasurements = params.get<bool>("verboseMeasurements", false);
    saveFrequency = saveFrequencyInTime / deltat;

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
    PetscPrintf(PETSC_COMM_WORLD, "finaltime = %e\n", finaltime);
    PetscPrintf(PETSC_COMM_WORLD, "initialtime = %e\n", initialtime);
    PetscPrintf(PETSC_COMM_WORLD, "deltat  = %e\n", deltat);
    PetscPrintf(PETSC_COMM_WORLD, "evolverType = %d\n", evolverType);

    // Action
    PetscPrintf(PETSC_COMM_WORLD, "mass = %e\n", mass);
    PetscPrintf(PETSC_COMM_WORLD, "lambda = %e\n", lambda);
    PetscPrintf(PETSC_COMM_WORLD, "gamma = %e\n", gamma);
    PetscPrintf(PETSC_COMM_WORLD, "H = %e\n", H);

    PetscPrintf(PETSC_COMM_WORLD, "seed = %d\n", seed);

    // Initialization
    PetscPrintf(PETSC_COMM_WORLD,
                "# Choose initFile=x to set coldStart to true\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Expected value of coldStart = %s\n",
                (coldStart ? "true" : "false"));
    PetscPrintf(PETSC_COMM_WORLD, "initFile = %s\n", initFile.c_str());

    PetscPrintf(PETSC_COMM_WORLD,
                "# Uncomment zero start to set the initial value to zero\n");
    PetscPrintf(PETSC_COMM_WORLD, "# zeroStart = %s\n",
                (zeroStart ? "true" : "false"));

    // Outputs
    PetscPrintf(PETSC_COMM_WORLD, "outputfiletag = %s\n",
                outputfiletag.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "saveFrequencyInTime = %e\n",
                saveFrequencyInTime);
    PetscPrintf(PETSC_COMM_WORLD, "# saveFrequency = %d\n", saveFrequency);
    PetscPrintf(PETSC_COMM_WORLD, "# Uncomment to print out every time step\n");
    PetscPrintf(PETSC_COMM_WORLD, "# verboseMeasurements = %s\n",
                (verboseMeasurements ? "true" : "false"));
  }
};

/////////////////////////////////////////////////////////////////////////

typedef struct {
  PetscScalar f[ModelAData::Ndof];
} o4_node;

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

  // Momentum used in backward step and eulerstep
  Vec phidot;

  //! Construct the grid and initialize the fields according to
  //! the configuration parameters in the input ModelAData structure
  ModelA(const ModelAData &in) : data(in) {

    PetscInt stencil_width = 1;
    DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                 DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, data.NX, data.NY,
                 data.NZ, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                 ModelAData::Ndof, stencil_width, NULL, NULL, NULL, &domain);

    DMSetFromOptions(domain);
    DMSetUp(domain);

    DMCreateGlobalVector(domain, &solution);
    VecDuplicate(solution, &previoussolution);
    VecDuplicate(solution, &phidot);

    // Setup the random number generation
    ModelARndm = std::make_unique<NoiseGenerator>(data.seed);
  }

  void finalize() {
    VecDestroy(&phidot);
    VecDestroy(&previoussolution);
    VecDestroy(&solution);
    DMDestroy(&domain);
  }

  //! Reads in a stored initial condition from Derek's file.
  //! This is a helper function for initialize
  PetscErrorCode loadFromDereksFile() {
    PetscViewer initViewer;
    PetscViewerHDF5Open(PETSC_COMM_WORLD, data.initFile.c_str(), FILE_MODE_READ,
                        &initViewer);
    PetscViewerSetFromOptions(initViewer);
    PetscObjectSetName((PetscObject)solution, "final_phi");
    PetscErrorCode ierr = VecLoad(solution, initViewer);
    CHKERRQ(ierr);
    PetscObjectSetName((PetscObject)solution, "o4fields");
    PetscViewerDestroy(&initViewer);
    return ierr;
  }

  //! Initialize the vector solution. If coldStart is false
  //! then read in the data from Derek's file. Otherwise fill with
  //! random numbers, or if zeroStart is true set to zero. Finally
  //! if a function is provided, f(x,y,z,l,params), this function will be used
  PetscErrorCode initialize(double (*func)(const double &x, const double &y, const double &z, const int &l, void *params)=0, void *params=0) {

    // Read in from a file
    if (!data.coldStart) {
      return loadFromDereksFile();
    }

    // Compute the lattice spacing
    PetscReal hx = data.hX();
    PetscReal hy = data.hY();
    PetscReal hz = data.hZ();

    // This Get a pointer to do the calculation
    o4_node ***u;
    DMDAVecGetArray(domain, solution, &u);

    // Get the Local Corner od the vector
    PetscInt i, j, k, l, xstart, ystart, zstart, xdimension, ydimension,
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
          for (l = 0; l < ModelAData::Ndof; l++) {
            if (data.zeroStart) {
              u[k][j][i].f[l] = 0.0;
            } else {
              if (func) {
                u[k][j][i].f[l] = func(x,y,z,l,params);
              } else {
                u[k][j][i].f[l] = ModelARndm->normal();
              }
            }
          }
        }
      }
    }
    DMDAVecRestoreArray(domain, solution, &u);
    return (0);
  }
};

#endif
