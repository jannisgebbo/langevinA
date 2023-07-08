#include <iomanip>
#include <iostream>
#include <sstream>
#include "measurer.h"

#ifndef MODELA_NO_HDF5
/////////////////////////////////////////////////////////////////////////

measurer_output_fasthdf5::measurer_output_fasthdf5(Measurer *in) : measure(in) {

  const auto &ahandler = measure->model->data.ahandler;

  std::string name;
  if (ahandler.eventmode) {
    std::stringstream namestream;
    namestream << ahandler.outputfiletag << "_" << std::setw(4)
               << std::setfill('0') << ahandler.current_event << ".h5";
    name = namestream.str();
  } else {
    name = ahandler.outputfiletag + ".h5";
  }

  if (ahandler.eventmode) {
    // Each event is stored in a separate file
    file_id = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    // Check if we are supposed to, and are able to, restart the measurements
    PetscBool restartflg = PETSC_FALSE;
    if (ahandler.restart && !ahandler.eventmode) {
      PetscTestFile(name.c_str(), '\0', &restartflg);
    }

    if (restartflg and !ahandler.eventmode) {
      // File exists and we are in restart mode,  so open it for readwrite
      file_id = H5Fopen(name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    } else {
      // File either doesn't exist or we are not in restart mode, create a new
      // file
      file_id =
          H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
  }

  std::array<size_t, 1> NN1{Measurer::NScalars};
  scalars = std::make_unique<ntuple<1>>(NN1, "phi", file_id);

  std::array<size_t, 2> NN2{Measurer::NObs, static_cast<size_t>(measure->N)};
  wallx = std::make_unique<ntuple<2>>(NN2, "wallx", file_id);
  wally = std::make_unique<ntuple<2>>(NN2, "wally", file_id);
  wallz = std::make_unique<ntuple<2>>(NN2, "wallz", file_id);

  std::array<size_t, 1> NN3{2} ;
  timeout = std::make_unique<ntuple<1>>(NN3, "timeout", file_id) ;

}

measurer_output_fasthdf5::~measurer_output_fasthdf5() { H5Fclose(file_id); }
void measurer_output_fasthdf5::update() { ; }

void measurer_output_fasthdf5::save(const std::string &what) {
  for (int k = 0; k < Measurer::NScalars; k++) {
    scalars->row[k] = measure->OAverage[k];
  }
  for (size_t i = 0; i < Measurer::NObs; i++) {
    for (size_t j = 0; j < measure->N; j++) {
      size_t k = wallx->at({i, j});
      wallx->row[k] = measure->sliceAveragesX[i][j];
      wally->row[k] = measure->sliceAveragesY[i][j];
      wallz->row[k] = measure->sliceAveragesZ[i][j];
    }
  } 
  timeout->row[0] = measure->model->data.atime.t() ; 
  timeout->row[1] = measure->model->data.mass() ;

  scalars->fill();
  wallx->fill();
  wally->fill();
  wallz->fill();
  timeout->fill() ;
}
#endif

/////////////////////////////////////////////////////////////////////////
//! Helper routine which opens the ASCII viewer with a
//! specific mode
PetscErrorCode PetscViewerASCIIOpenMode(MPI_Comm comm, const char *name,
                                        PetscFileMode mode,
                                        PetscViewer *viewer) {
  PetscViewerCreate(PETSC_COMM_SELF, viewer);
  PetscViewerSetType(*viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(*viewer, mode);
  PetscViewerFileSetName(*viewer, name);
  return (0);
}

measurer_output_txt::measurer_output_txt(Measurer *in) : measure(in) {

  const auto &ahandler = measure->model->data.ahandler;
  std::string name(ahandler.outputfiletag + "_averages.txt");
  PetscInt ierr = PetscViewerASCIIOpenMode(
      PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE, &averages_asciiviewer);
  CHKERRV(ierr);
}

measurer_output_txt::~measurer_output_txt() {
  PetscViewerDestroy(&averages_asciiviewer);
}

void measurer_output_txt::save(const std::string &what) {

  if (what == "phi") {
    for (int iO = 0; iO < measure->NScalars; iO++) {
      PetscViewerASCIIPrintf(averages_asciiviewer, "%22.15e ",
                             measure->OAverage[iO]);
    }
    PetscViewerASCIIPrintf(averages_asciiviewer, "\n");
  }
}
