#include <memory>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <vector>

#include "Stepper.h"
#include "NoiseGenerator.h"
#include "ModelA.h"

// Homemade parser
#include "parameterparser/parameterparser.h"

// Measurer, where the Petsc are included
#include "measurer.h"

int main(int argc, char **argv) {
  // Initialization
  PetscErrorCode ierr;
  std::string help = "Model A. Call ./ModelA-Beuler.exe input=input.in with "
                     "input.in your input file.";

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
  ModelA model(inputdata);
  model.initialize() ;

  // initialize the measurments and measure the initial condition
  Measurer measurer(&model);
  measurer.measure(&model.solution, &model.phidot);

  // Initialize the stepper
  std::unique_ptr<Stepper> step ; 

  switch (model.data.evolverType) {
  case 1:
    step = std::make_unique<BackwardEuler>(model) ;
    break;
  case 2:
    step = std::make_unique<ForwardEuler>(model) ;
    break;
  }

  PetscInt steps = 1;
  // Thsi is the loop for the time step
  for (PetscReal time = model.data.initialtime; time < model.data.finaltime;
       time += model.data.deltat) {
    // Copy the solution 
    VecCopy(model.solution, model.previoussolution);

    // generate the noise
    ModelARndm->fillVec(model.noise) ;

    step->step(model.data.deltat) ;

    // mesure the solution
    if (steps % model.data.saveFrequency == 0) {
      measurer.measure(&model.solution, &model.phidot);
      // Print some information to not get bored during the running:
      PetscPrintf(PETSC_COMM_WORLD, "Timestep %D: step size = %g, time = %g\n",
                steps, (double)model.data.deltat, (double)time);
    }

    
    steps++;
  }

  // Destroy everything
  step->finalize() ;
  measurer.finalize();
  model.finalize() ;
  return PetscFinalize();
}


