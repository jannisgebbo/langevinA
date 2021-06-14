#include "Stepper.h"
#include "ModelA.h"
// Homemade parser
#include "parameterparser/parameterparser.h"

int main(int argc, char **argv) {
  // Initialization
  PetscErrorCode ierr;
  std::string help = "Tests a simple matrix inversion";
  ierr = PetscInitialize(&argc, &argv, (char *)0, help.c_str());
  if (ierr) {
    return ierr;
  }

  // Read the data form command line
  FCN::ParameterParser params(argc, argv);
  ModelAData inputdata(params);

  // If -quit flag, then just stop before we do anything but gather inputs.
  PetscBool quit = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-quit", &quit, NULL);
  CHKERRQ(ierr);
  if (quit) {
    return PetscFinalize();
  }


  // allocate the grid and initialize
  inputdata.zeroStart = true;
  ModelA model(inputdata);
  model.initialize() ;
  auto step = std::make_unique<BackwardEuler>(model) ;

  // Copy the solution 
  VecCopy(model.solution, model.previoussolution);
  // Try inversion
  step->step(model.data.deltat) ;

  // Destroy everything
  step->finalize() ;
  model.finalize();
  return PetscFinalize();
}
