

// This simple class is used to printout the fields phi at regular intervals
// into an hdf5 file
class plotter {

public:
  plotter(const std::string &filename) {
    std::string name = filename + ".h5";
    PetscViewerHDF5Open(PETSC_COMM_WORLD, name.c_str(), FILE_MODE_WRITE,
                        &H5viewer);
  }

  void plot(Vec V, const std::string &name) {
    PetscObjectSetName((PetscObject)V, name.c_str());
    VecView(V, H5viewer);
  }
  void finalize() { PetscViewerDestroy(&H5viewer); }

private:
  PetscViewer H5viewer;
};
