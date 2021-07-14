#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "ModelA.h"
#include "NoiseGenerator.h"
#include "Stepper.h"
#include "make_unique.h"

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
  model.initialize();

  // initialize the measurments and measure the initial condition
  Measurer measurer(&model);
  measurer.measure(&model.solution);

  // Initialize the stepper
  std::unique_ptr<Stepper> step;

  switch (model.data.evolverType) {
  case 1:
    step = make_unique<BackwardEuler>(model);
    break;
  case 2:
    step = make_unique<ForwardEuler>(model);
    break;
  case 3:
    step = make_unique<SemiImplicitBEuler>(model);
    break;
  case 4:
    step = make_unique<EulerLangevinHB>(model);
    break;
  case 5:
    step = make_unique<IdealLF>(model);
    break;
  case 6:
    step = make_unique<LFHBSplit>(model, model.data.deltatHB);
    break;
  }

  PetscInt steps = 1;
  PetscLogEvent measurements;
  PetscLogEventRegister("Measurements", 0, &measurements);
  // Thsi is the loop for the time step
  for (PetscReal time = model.data.initialtime; time < model.data.finaltime;
       time += model.data.deltat) {
    // Copy the solution
    VecCopy(model.solution, model.previoussolution);

    step->step(model.data.deltat);

    // measure the solution
    PetscLogEventBegin(measurements, 0, 0, 0, 0);
    if (steps % model.data.saveFrequency == 0) {
      measurer.measure(&model.solution);
      // Print some information to not get bored during the running:
      PetscPrintf(PETSC_COMM_WORLD, "Timestep %D: step size = %g, time = %g\n",
                  steps, (double)model.data.deltat, (double)time);
    }
    PetscLogEventEnd(measurements, 0, 0, 0, 0);

    steps++;
  }

  // Destroy everything
  step->finalize();
  measurer.finalize();
  model.finalize();
  return PetscFinalize();
}
