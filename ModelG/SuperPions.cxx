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

// Measurer, where the Petsc are included
#include "measurer.h"

int main(int argc, char **argv) {

  // Initialization of PETSc universe
  PetscErrorCode ierr;
  std::string help =
      "Usage: \n\n\t ./SuperPions.exe -input input.json [options]\n\n";
  ierr = PetscInitialize(&argc, &argv, (char *)0, help.c_str());
  if (ierr) {
    return ierr;
  }

  // Open the input file and parse the inputs into the ModelAData
  char filename[PETSC_MAX_PATH_LEN] = "";
  ierr = PetscOptionsGetString(NULL, NULL, "-input", filename, sizeof(filename),
                               NULL);
  Json::Value inputs;
  std::ifstream ifs(filename);
  if (ifs) {
    ifs >> inputs;
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unable to open input file %s. Aborting...\n",
                filename);
    return PetscFinalize();
  }

  // Digest the inputs, some fields may be modified on ouptut
  ModelAData inputdata(inputs);

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

  // Initialize the stepper
  std::unique_ptr<Stepper> step;

  std::string &s = model.data.evolverType;
  if (s == "EulerLangevinHB") {
    // Just take the Langevin updates of scalars
    step = make_unique<EulerLangevinHB>(model);
  } else if (s == "IdealPV2") {
    // Just Ideal Steps
    step = make_unique<IdealPV2>(model);
  } else if (s == "ModelGChargeHB") {
    // Just take a langevin step for the charges
    step = make_unique<ModelGChargeHB>(model);
  } else if (s == "PV2HBSplit11") {
    // ABC
    step = make_unique<PV2HBSplit>(model);
  } else if (s == "PV2HBSplit23") {
    // ABB,ABB,ABB,C
    std::array<unsigned int, 2> s = {2, 3};
    step = std::make_unique<PV2HBSplit>(model, s);
  }

  PetscInt steps = 0;
  PetscLogEvent measurements, stepmonitor;
  PetscLogEventRegister("Measurements", 0, &measurements);
  PetscLogEventRegister("Steps", 0, &stepmonitor);

  // This is the loop for the time step. It goes
  //
  // PRE-Loop
  //
  // Loop:
  // measure&step, step, step,  measure&step, step step,
  //
  // POST-Loop
  //
  // Thus restarting the program will give the same data
  // stream, as if there was no interuption.
  PetscReal time = model.data.initialtime;
  const double tiny = 1.e-10;
  while (time < model.data.finaltime - tiny) {

    // measure the solution
    if (steps % model.data.saveFrequency == 0) {
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure(&model.solution);
      // Print some information to not get bored during the running:
      PetscPrintf(PETSC_COMM_WORLD,
                  "Timestep %D: step size = %g, time = %g, final = %g\n", steps,
                  (double)model.data.deltat, (double)time,
                  (double)model.data.finaltime);
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
    }

    // Copy the solution to have the current and previous step in memory
    VecCopy(model.solution, model.previoussolution);

    // Do the actual steps
    PetscLogEventBegin(stepmonitor, 0, 0, 0, 0);
    step->step(model.data.deltat);
    PetscLogEventEnd(stepmonitor, 0, 0, 0, 0);

    steps++;
    time += model.data.deltat;
  }

  // Destroy everything
  step->finalize();
  measurer.finalize();
  model.finalize();
  return PetscFinalize();
}
