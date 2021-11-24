#ifndef MODELASTRUCT
#define MODELASTRUCT

#include "NoiseGenerator.h"
#include "parameterparser/parameterparser.h"
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
  static const PetscInt Nphi = 4;
  static const PetscInt NA = 3;
  static const PetscInt NV = 3;
  static const PetscInt Ndof = Nphi + NA + NV;

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
  PetscReal diffusion = 0.3333333;
  PetscReal chi = 1.1;
  PetscReal sigma() const {return diffusion*chi;}
  PetscReal D() const {return diffusion;}

  // random seed
  PetscInt seed = 10;

  // Options controlling the initial condition
  bool restart = false;

  // Options controlling the output. The outputfiletag
  // labells the run. All output files are tag_foo.txt, or tag_bar.h5
  std::string outputfiletag = "o4output";
  PetscReal saveFrequencyInTime;
  PetscInt saveFrequency;
  bool verboseMeasurements;

  ModelAData(const Json::Value &params) {
    // Lattice. By default, NY=NX and NZ=NX.
    NX = params["NX"].asInt();
    NY = NX;
    NZ = NX;

    // By default, dx=dy=dz=1, namely LX=NX, LY=NY, LZ=NZ.
    LX = params.get("LX", NX).asInt();
    LY = LX;
    LZ = LX;

    // Time Stepping
    finaltime = params["finaltime"].asDouble();
    initialtime = params["initialtime"].asDouble();
    deltat = params["deltat"].asDouble();
    evolverType = params["evolverType"].asInt();

    // Action
    mass = params["mass"].asDouble();
    lambda = params["lambda"].asDouble();
    gamma = params["gamma"].asDouble();
    H = params["H"].asDouble();
    diffusion = params.get("diffusion", 1./3.).asDouble();
    chi = params.get("chi", 1.35).asDouble();

    seed = (PetscInt)params["seed"].asInt();

    // Control initialization
    restart = params.get("restart", false).asBool() ;

    // Control outputs
    outputfiletag = params.get("outputfiletag", "o4output").asString();
    saveFrequencyInTime = params["saveFrequencyInTime"].asDouble();
    verboseMeasurements = params.get("verboseMeasurements", false).asBool();
    saveFrequency = saveFrequencyInTime / deltat;

    // In order to work in restart mode final time should be an integral number of saveFrequency
    // So that the first action, upon reloading is to save the datastream.
    finaltime = static_cast<int>(finaltime/(deltat * saveFrequency)) * deltat * saveFrequency ;

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
    PetscPrintf(PETSC_COMM_WORLD, "diffusion = %e\n", diffusion);
    PetscPrintf(PETSC_COMM_WORLD, "chi = %e\n", chi);

    PetscPrintf(PETSC_COMM_WORLD, "seed = %d\n", seed);

    // Initialization
    PetscPrintf(PETSC_COMM_WORLD, "restart = %s\n", (restart ? "true" : "false"));

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
    ModelARndm = make_unique<NoiseGenerator>(data.seed);
    if (data.restart) { 
      ModelARndm->read(data.outputfiletag) ;
    }

    // Printout store the rank
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }

  void finalize() {
    ModelARndm->write(data.outputfiletag) ;
    write(data.outputfiletag) ;
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
    PetscBool flg ;
    PetscErrorCode ierr = PetscTestFile(fname.c_str(), '\0' ,  &flg) ;
    if (!flg) {
      throw(std::string("Unable to open file in restart mode with filename = ") + fname ) ;
    }
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fname.c_str(), FILE_MODE_READ,
                        &initViewer);
    CHKERRQ(ierr) ; 
    PetscViewerSetFromOptions(initViewer);
    PetscObjectSetName((PetscObject)solution, "o4fields");
    ierr = VecLoad(solution, initViewer);
    CHKERRQ(ierr) ;
    PetscViewerDestroy(&initViewer);
    return ierr ;
#else
    throw("ModelA::read Unable to load from file without HDF5 support. \n");
    return 0;
#endif
  }
  
  //! Reads in a stored initial condition from Derek's file.
  //! This is a helper function for initialize
  PetscErrorCode write(const std::string fnamein) {
    std::string fname = fnamein + "_save.h5" ;
#ifndef MODELA_NO_HDF5
    PetscViewer initViewer;
    PetscErrorCode ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fname.c_str(), FILE_MODE_WRITE,
                        &initViewer);
    CHKERRQ(ierr) ;
    PetscViewerSetFromOptions(initViewer);
    PetscObjectSetName((PetscObject)solution, "o4fields");
    ierr = VecView(solution, initViewer);
    CHKERRQ(ierr) ;
    PetscViewerDestroy(&initViewer);
    return 0;
#else
    PetscPrintf("ModelA::write Unable to write to a file wihtout HDF5 support\n") ;
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

    if (data.restart) {
      try {
        read(data.outputfiletag) ;
        return (0);
      } catch(const std::string &error) {
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
