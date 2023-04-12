#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <vector>

#include "ModelA.h"
#include "NoiseGenerator.h"
#include "Stepper.h"
#include "make_unique.h"

// Measurer, where the Petsc are included
#include "measurer.h"

void run_event(ModelA* const model,Stepper* const step) 
{

  const auto &ahandler = model->data.ahandler ;
  auto &atime = model->data.atime ;

  atime.reset() ;

  // Thermalize the state in memory at the initial time ;
  int nsteps  = static_cast<int>(ahandler.thermalization_time / atime.dt()) ;
  PetscPrintf(PETSC_COMM_WORLD, "Thermalizing event %D\n", ahandler.current_event); 
  for (int i = 0 ; i < nsteps ; i++) {
    step->step(atime.dt()) ;
    PetscPrintf(PETSC_COMM_WORLD,
                "Thermalizing Event/Timestep %D/%D: step size = %g, time = %g, nsteps to thermalize = %D \n", ahandler.current_event, i, (double)atime.dt(),
                (double)atime.t(), nsteps);
  }

  PetscInt steps = 0;
  PetscLogEvent measurements, stepmonitor, saving;
  PetscLogEventRegister("Measurements", 0, &measurements);
  PetscLogEventRegister("Saving the fields", 0, &saving);
  PetscLogEventRegister("Steps", 0, &stepmonitor);

  // Initialize the measurments and measure the initial condition for this
  // event.  The data for each event is stored in a separate file
  Measurer measurer(model);

  // Start the loop
  const double tiny = 1.e-10;
  while (atime.t() < atime.tfinal() - tiny) {

    // measure the solution every saveFrequency
    if (steps % ahandler.saveFrequency == 0) {
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure(&model->solution);
      // Print some information to not get bored during the running:
      PetscPrintf(PETSC_COMM_WORLD,
                  "Event/Timestep %D/%D: step size = %g, time = %g, final = %g\n", ahandler.current_event, steps,
                  (double)atime.dt(), (double)atime.t(), (double)atime.tfinal());
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
    }

    // Write the solution to tape if writeFrequency > 0. This is used for plotting.
    if(ahandler.writeFrequency > 0 and steps % ahandler.writeFrequency == 0) {
      PetscLogEventBegin(saving, 0, 0, 0, 0);
      std::ostringstream tString;
      tString << std::setprecision(4) <<"_t_" << atime.t();
      model->write(ahandler.outputfiletag + tString.str());
      PetscLogEventEnd(saving, 0, 0, 0, 0);
    }

    // Do the actual steps
    PetscLogEventBegin(stepmonitor, 0, 0, 0, 0);
    step->step(atime.dt());
    PetscLogEventEnd(stepmonitor, 0, 0, 0, 0);

    // Increment the clock
    steps++;
    atime += atime.dt();
  }
  measurer.finalize();

}


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

  // Read in the initial conditions or initialize to zero 
  model.initialize() ;

  // Construct the stepper 
  std::unique_ptr<Stepper> step; 
  auto &etype = inputdata.ahandler.evolverType ;
  if (etype == "PV2HBSplit23") {
     std::array<unsigned int, 2> s = {2, 3};
     step = std::make_unique<PV2HBSplit>(model, s);
  } else if (etype == "PV2HBSplit23NoDiffuse") {
     std::array<unsigned int, 2> s = {2, 3};
     const bool nodiffuse = true ;
     step = std::make_unique<PV2HBSplit>(model, s, nodiffuse);
  } else if (etype == "PV2HBSplit23OnlyDiffuse") {
     std::array<unsigned int, 2> s = {2, 3};
     const bool nodiffuse = false;
     const bool onlydiffuse = true ;
     step = std::make_unique<PV2HBSplit>(model, s, nodiffuse, onlydiffuse);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unrecognized stepper type %s. Aborting...\n",
                etype.c_str());
    return PetscFinalize();
  }

  auto &ahandler = model.data.ahandler ;
  if (ahandler.eventmode) { 
    for (int i = 0 ;  i < ahandler.nevents ; i++ )  {
      run_event(&model, step.get()) ;
      ahandler.current_event++ ;
    }
  } else  {
    run_event(&model, step.get()) ;
  }

  // Destroy everything
  step->finalize();
  model.finalize();
  return PetscFinalize();
}
