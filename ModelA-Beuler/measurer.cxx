#include "measurer.h"

#ifndef MODELA_NO_HDF5
measurer_output_hdf5::measurer_output_hdf5(Measurer *in) : measure(in) {

  ModelA *model = measure->model;
  // Create temporary petsc arrays for holding scalars
  VecCreateSeq(PETSC_COMM_SELF, measure->NScalars, &vecSaverScalars);
  VecCreateSeq(PETSC_COMM_SELF, measure->N, &vecSaverCors);

  // Set the parameters controlling the output
  verbosity = model->data.verboseMeasurements;
  std::string name(model->data.outputfiletag + ".h5");
  PetscViewerHDF5Open(PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE, &viewer);

  PetscViewerSetFromOptions(viewer);
  PetscViewerHDF5SetTimestep(viewer, model->data.initialtime);
}

measurer_output_hdf5::~measurer_output_hdf5() {
  VecDestroy(&vecSaverCors);
  VecDestroy(&vecSaverScalars);
  PetscViewerDestroy(&viewer);
};

void measurer_output_hdf5::update() {
  PetscViewerHDF5IncrementTimestep(viewer);
}

void measurer_output_hdf5::save(const std::string &what) {
  std::string tmp;
  for (PetscInt i = 0; i < measure->NObs; ++i) {
    tmp = what + "_" + std::to_string(i);
    saveCorLike(measure->sliceAveragesX[i], "wallX_" + tmp);
    if (verbosity)
      saveCorLike(measure->sliceAveragesY[i], "wallY_" + tmp);
    if (verbosity)
      saveCorLike(measure->sliceAveragesZ[i], "wallZ_" + tmp);
    if (verbosity)
      saveCorLike(measure->isotropicWallToWallCii[i], "Cii_" + tmp);
  }
  saveScalarsLike(measure->OAverage, what);
}
// Save instance of scalars like object into a hdf5 file
void measurer_output_hdf5::saveScalarsLike(std::vector<PetscScalar> &arr,
                                           const std::string &name) {

  // for indixing the petsc vectors where these will be saved
  std::vector<PetscInt> vecScalarsIndex;
  for (PetscInt i = 0; i < measure->NScalars; ++i) {
    vecScalarsIndex.emplace_back(i);
  }

  PetscObjectSetName((PetscObject)vecSaverScalars, name.c_str());

  VecSetValues(vecSaverScalars, measure->NScalars, vecScalarsIndex.data(),
               arr.data(), INSERT_VALUES);

  VecAssemblyBegin(vecSaverScalars);
  VecAssemblyEnd(vecSaverScalars);
  VecView(vecSaverScalars, viewer);
}
// Save instance of a correlator like object into a hdf5 file
void measurer_output_hdf5::saveCorLike(std::vector<PetscScalar> &arr,
                                       const std::string &name) {

  // For indexing the Petsc objects where these will be saved
  std::vector<PetscInt> vecCorsIndex;
  for (PetscInt i = 0; i < measure->N; ++i) {
    vecCorsIndex.emplace_back(i);
  }

  PetscObjectSetName((PetscObject)vecSaverCors, name.c_str());
  VecSetValues(vecSaverCors, measure->N, vecCorsIndex.data(), arr.data(),
               INSERT_VALUES);

  VecAssemblyBegin(vecSaverCors);
  VecAssemblyEnd(vecSaverCors);
  VecView(vecSaverCors, viewer);
}

#endif

/////////////////////////////////////////////////////////////////////////
//! Helper routine for o4_run_create, which opens the ASCII viewer with a
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

  std::string name(measure->model->data.outputfiletag + "_averages.txt");
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
