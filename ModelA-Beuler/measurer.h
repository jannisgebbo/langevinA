#ifndef MEASURER
#define MEASURER

#include "modelAStruct.h"


class Measurer{
public:

  Measurer(void *ptr)
  {
    global_data* user =(global_data*) ptr;
    da=user->da;
    N = user->model.NX;
    Ndof = user->model.Ndof;
    if (user->model.NX != user->model.NY || user->model.NX != user->model.NZ) {
        PetscPrintf(
                    PETSC_COMM_WORLD,
                    "Nx, Ny, and Nz must be equal for the correlation analysis to work");
        throw("Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }
    int nObs = 4;
    VecCreateSeq(PETSC_COMM_SELF,nObs, &vecSaverScalars);
    VecCreateSeq(PETSC_COMM_SELF,N, &vecSaverCors);

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if(rank == 0) {
      std::string name(user->filename + ".h5");
      PetscViewerHDF5Open(PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE, &viewer);
      PetscViewerSetFromOptions(viewer);
      PetscViewerHDF5SetTimestep(viewer, user->model.initialtime);
    }

    for(PetscInt i =0; i<N; ++i){
      vecCorsIndex.emplace_back(i);
    }
    for(PetscInt i =0; i<nObs; ++i){
      vecScalarsIndex.emplace_back(i);
    }
  }

  virtual ~Measurer(){
  }

  void finalize()
  {
    VecDestroy(&vecSaverCors);
    VecDestroy(&vecSaverScalars);
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if(rank == 0) PetscViewerDestroy(&viewer);
  }

  void measure(Vec *solution)
  {

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    computeSliceAverage(solution);
    if(rank == 0){
      computeDerivedObs();
      save();
    }
  }


private:



    void computeSliceAverage(Vec *solution)
    {
      //Get the local information and store in info

      Vec localU;
      DMGetLocalVector(da,&localU);
      //take the global vector U and distribute to the local vector localU
      DMGlobalToLocalBegin(da,*solution,INSERT_VALUES,localU);
      DMGlobalToLocalEnd(da,*solution,INSERT_VALUES,localU);

      //From the vector define the pointer for the field phi
      o4_node ***phi;
      DMDAVecGetArrayRead(da,localU,&phi);


      // OX0[i] will contain the average of phi0 at wall x=i.
      // And analagous vectors for the wall averages in the y and z directions

      sliceAveragesX = std::vector<std::vector<PetscScalar>>(Ndof, std::vector<PetscScalar>(N));
      sliceAveragesY = std::vector<std::vector<PetscScalar>>(Ndof, std::vector<PetscScalar>(N));
      sliceAveragesZ = std::vector<std::vector<PetscScalar>>(Ndof, std::vector<PetscScalar>(N));
      auto sliceAveragesLocalX = sliceAveragesX;
      auto sliceAveragesLocalY = sliceAveragesY;
      auto sliceAveragesLocalZ = sliceAveragesZ;

      // Get the ranges
      PetscInt ixs, iys, izs, nx, ny, nz;
      DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

      for (int k = izs; k < izs + nz; k++) {
          for (int j = iys; j < iys + ny; j++) {
              for (int i = ixs; i < ixs + nx; i++) {
                  for (int l = 0; l < Ndof; l++) {
                      sliceAveragesLocalX[l][i] += phi[k][j][i].f[l];
                      sliceAveragesLocalY[l][j] += phi[k][j][i].f[l];
                      sliceAveragesLocalZ[l][k] += phi[k][j][i].f[l];
                  }
              }
          }
      }

      for(int l = 0; l<Ndof; ++l){
        for(int i = 0; i<N; ++i){
          sliceAveragesLocalX[l][i] /= PetscReal(N * N);
          sliceAveragesLocalY[l][i] /= PetscReal(N * N);
          sliceAveragesLocalZ[l][i] /= PetscReal(N * N);
        }
      }
      // Retstore the array
      DMDAVecRestoreArray(da, localU, &phi);

      // Bring all the data x data into one
      for (int l = 0; l < Ndof; l++) {
        MPI_Reduce(sliceAveragesLocalX[l].data(), sliceAveragesX[l].data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
        MPI_Reduce(sliceAveragesLocalY[l].data(), sliceAveragesY[l].data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
        MPI_Reduce(sliceAveragesLocalZ[l].data(), sliceAveragesZ[l].data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
      }
    }

    void computeDerivedObs()
    {
      isotropicWallToWallCii = std::vector<std::vector<PetscScalar>>(Ndof, std::vector<PetscScalar>(N));
      OAverage = std::vector<PetscScalar>(Ndof);
      for (int l = 0; l < Ndof; l++) {
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
              int jj = (i + j) % N; // Circulate the index, if i=N-1 j=1, jj=0

              isotropicWallToWallCii[l][j] +=
              (sliceAveragesX[l][i] * sliceAveragesX[l][jj] + sliceAveragesY[l][i] * sliceAveragesY[l][jj] + sliceAveragesZ[l][i] * sliceAveragesZ[l][jj]) /3.;
          }
          OAverage[l] += sliceAveragesX[l][i];
        }
      }
      for (int l = 0; l < Ndof; l++) {
          OAverage[l] /= PetscReal(N);
      }
    }


    void save()
    {
      std::string phi;
      for(PetscInt i = 0 ; i< Ndof ; ++i ){
        phi = "phi_"+std::to_string(i);
        saveCorLike(sliceAveragesX[i], "wallX_"+ phi);
        saveCorLike(sliceAveragesY[i], "wallY_"+ phi);
        saveCorLike(sliceAveragesZ[i], "wallZ_"+ phi);

        saveCorLike(isotropicWallToWallCii[i], "Cii_" + phi);

      }
      saveScalarsLike(OAverage, "phi");
      PetscViewerHDF5IncrementTimestep(viewer);

    }



  void saveCorLike(std::vector<PetscScalar>& arr, std::string name)
  {
    PetscObjectSetName((PetscObject) vecSaverCors, name.c_str());

    VecSetValues(vecSaverCors,N,vecCorsIndex.data(),arr.data(),INSERT_VALUES);

    VecAssemblyBegin(vecSaverCors);
    VecAssemblyEnd(vecSaverCors);
    VecView(vecSaverCors, viewer);
  }

  void saveScalarsLike(std::vector<PetscScalar>& arr, std::string name)
  {
    PetscObjectSetName((PetscObject) vecSaverScalars, name.c_str());

    VecSetValues(vecSaverScalars,Ndof,vecScalarsIndex.data(),arr.data(),INSERT_VALUES);

    VecAssemblyBegin(vecSaverScalars);
    VecAssemblyEnd(vecSaverScalars);
    VecView(vecSaverScalars, viewer);
  }


    DM da;
    PetscInt N;
    PetscInt Ndof;
    Vec vecSaverCors;
    std::vector<PetscInt> vecCorsIndex;
    Vec vecSaverScalars;
    std::vector<PetscInt> vecScalarsIndex;



    std::vector<
      std::vector<PetscScalar>
    > sliceAveragesX; //First dimension O4 component, last dimension slice number.
    std::vector<
      std::vector<PetscScalar>
    > sliceAveragesY; //First dimension O4 component, last dimension slice number.
    std::vector<
      std::vector<PetscScalar>
    > sliceAveragesZ; //First dimension O4 component, last dimension slice number.


    std::vector<
      std::vector<PetscScalar>
    > isotropicWallToWallCii;

    std::vector<PetscScalar> OAverage;

    // Viewer
    PetscViewer viewer;

};



#endif
