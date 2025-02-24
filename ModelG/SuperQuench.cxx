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
#include "measurer_output.h"



void initializ_event(ModelA *model) 
{

}
void run_event(ModelA* const model,Stepper* const step) 
{

  const auto &ahandler = model->data.ahandler ;
  auto &atime = model->data.atime ;
  atime.reset() ;

  initialize_event(model) ;

  // Set up logging for PETSc so we can find out how much time 
  // each part takes
  PetscInt steps = 0;
  PetscLogEvent measurements, stepmonitor, saving;
  PetscLogEventRegister("Measurements", 0, &measurements);
  PetscLogEventRegister("Saving the fields", 0, &saving);
  PetscLogEventRegister("Steps", 0, &stepmonitor);

  // Set filename for the output
  std::string filename;
  if (ahandler.eventmode) {
    std::stringstream namestream;
    namestream << ahandler.outputfiletag << "_" << std::setw(4)
               << std::setfill('0') << ahandler.current_event << ".h5";
    filename = namestream.str();
  } else {
    filename = ahandler.outputfiletag + ".h5";
  }
  // Set file access
  PetscFileMode file_access = FILE_MODE_WRITE ;
  if (ahandler.restart) {
    file_access = FILE_MODE_APPEND;
  }  
  // Open the file and create the measurement object
  Measurer measurer(model) ;
  measurer_output_fasthdf5 measurer_output(&measurer, filename, file_access);

  // Start the loop
  const double tiny = 1.e-10;
  while (atime.t() < atime.tfinal() - tiny) {

    // measure the solution every saveFrequency
    if (steps % ahandler.saveFrequency == 0) {
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure(&model->solution);
      measurer_output.save() ;
      PetscPrintf(PETSC_COMM_WORLD,
                  "Event/Timestep %D/%D: step size = %g, time = %g, final = %g\n", ahandler.current_event, steps,
                  (double)atime.dt(), (double)atime.t(), (double)atime.tfinal());
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
    }

    // Do the actual steps
    PetscLogEventBegin(stepmonitor, 0, 0, 0, 0);
    step->step(atime.dt());
    PetscLogEventEnd(stepmonitor, 0, 0, 0, 0);

    // Increment the clock
    steps++;
    atime += atime.dt();
  }
}


int main(int argc, char **argv) {

  // Initialization of PETSc universe
  PetscErrorCode ierr;
  std::string help =
      "Usage: \n\n\t ./SuperQuench.exe -input input.json [options]\n\n";
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

  // Construct the stepper 
  std::unique_ptr<Stepper> step; 
  step = make_unique<IdealPV2>(model);

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
