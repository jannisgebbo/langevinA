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


// This is an example of how to initialize the data in the grid. This is a
// simple example where the charges are gaussian random numbers. The charges
// are normalized so that the total charge is zero.
void initialize_event(ModelA *model) 
{
  // ModelAData is lightweight and contains all the data for ModelA
  ModelAData &data = model->data ;
  // Compute the lattice spacing extracting the information from the data
  // PetscReal hx = data.hX();
  // PetscReal hy = data.hY();
  // PetscReal hz = data.hZ();

  // data_node ***u
  //
  // This Get a pointer to do the calculation. 
  // Here we are viewing the grid as a three dimensional grid of data_nodes
  //
  // G_node ***u;
  //
  // Alternatively one could use G_Node ***u which would view the grid as a 
  // three dimensional grid of G_Nodes (which divides up the fields in phi, V,
  // and A), see ModelA.h for more information.
  // 
  // Petscalar ****u
  //
  // Still alternatively one coude use DMDAVecGetArrayDOF(model->domain,
  // model->solution, &u); whichwould view the grid as a four dimensional grid
  // of PetscScalars.
  data_node ***u;
  DMDAVecGetArray(model->domain, model->solution, &u);

  // This is the Petsc way of doing things
  PetscInt i, j, k, L, xstart, ystart, zstart, xdimension, ydimension,
      zdimension;

  DMDAGetCorners(model->domain, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // We are going initialize the grid with the charges being gaussian random
  // numbers. The charges are normalized so that the total charge is zero. 
  std::vector<PetscScalar> charge_sum_local(ModelAData::Ndof, 0.);
  std::vector<PetscScalar> charge_sum(ModelAData::Ndof, 0.);

  PetscScalar chi = data.acoefficients.chi ;
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {
        for (L = 0; L < ModelAData::Ndof; L++) {
          // Dont update the phi components
          if (L < ModelAData::Nphi) {
            continue;
          }

          // Generate gaussian random numbers for charges
          u[k][j][i].x[L] = sqrt(chi) * ModelARndm->normal();

          // Accumulate the total charge in a Buffer
          charge_sum_local[L] += u[k][j][i].x[L] ;
        }
      }
    }
  }
    // Find the total charge
  MPI_Allreduce(charge_sum_local.data(), charge_sum.data(), ModelAData::Ndof,
                MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

  // Subtract the the total charge from the grid. This assume the lattice
  // spacing is 1
  PetscScalar V = data.NX * data.NY * data.NX;
  for (k = zstart; k < zstart + zdimension; k++) {
    for (j = ystart; j < ystart + ydimension; j++) {
      for (i = xstart; i < xstart + xdimension; i++) {
        for (L = 0; L < ModelAData::Ndof; L++) {
          if (L < ModelAData::Nphi) {
            continue;
          }
          u[k][j][i].x[L] -= charge_sum[L] / V;
        }
      }
    }
  }

  DMDAVecRestoreArray(model->domain, model->solution, &u);

}

// This is the main loop of the program. It is a simple loop that steps the
// solution forward in time until the final time is reached. The data is analyzed
// and saved every saveFrequency steps.
void run_event(ModelA* const model,Stepper* const step) 
{

  const auto &ahandler = model->data.ahandler ;
  auto &atime = model->data.atime ;
  atime.reset() ;

  initialize_event(model) ;
  // Write the grid to a file so we can see the initial conditions
  // The "outputfiletag" is the base name of the file. The grid will be
  // written to a file with the name outputfiletag_save.h5. The _save.h5
  // is added by the write function.
  model->write(ahandler.outputfiletag + "_grid") ;

  // Set up logging for PETSc so we can find out how much time 
  // each part takes
  PetscInt steps = 0;
  PetscLogEvent measurements, stepmonitor, saving;
  PetscLogEventRegister("Measurements", 0, &measurements);
  PetscLogEventRegister("Saving the fields", 0, &saving);
  PetscLogEventRegister("Steps", 0, &stepmonitor);

  // Set filename for the hdf5 output of measurements
  std::string filename;
  if (ahandler.eventmode) {
    std::stringstream namestream;
    namestream << ahandler.outputfiletag << "_" << std::setw(4)
               << std::setfill('0') << ahandler.current_event << ".h5";
    filename = namestream.str();
  } else {
    filename = ahandler.outputfiletag + ".h5";
  }
  // Set file access mode for the hdf5 output of measurements
  PetscFileMode file_access = FILE_MODE_WRITE ;

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
                  "Event/Timestep %d/%d: step size = %g, time = %g, final = %g\n", ahandler.current_event, steps,
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

  // Digest the inputs, some fields may be modified on ouptut.
  // If you need to add more parameters add them to ModelAData
  ModelAData inputdata(inputs);

  // If -quit flag, then just stop before we do anything but gather inputs.
  PetscBool quit = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-quit", &quit, NULL);
  CHKERRQ(ierr);
  if (quit) {
    return PetscFinalize();
  }

  // allocate the grid 
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
