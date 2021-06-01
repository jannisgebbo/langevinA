#ifndef NOISEGENERATOR
#define NOISEGENERATOR

#include <random>

template<class RNGType>
class NoiseGenerator{
public :

NoiseGenerator(int baseSeed, DM& da): //PETSC indexing
normDist(0.0, 1.0)
{
  //PetscInt     Nx, Ny, Nz, npx, npy, npz, ndof, xstart, ystart, zstart,xdimension,ydimension,zdimension;
  //This function get the information of the global dimension of the grid.
  //DMDAGetInfo(da,PETSC_IGNORE, &Nx,&Ny,&Nz,&npx,&npy,&npz,&ndof,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  //DMDAGetCorners(da,&xstart,&ystart,&zstart,&xdimension,&ydimension,&zdimension);
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  //std::vector<std::uint32_t> seedArr(ndof * xdimension * ydimension * zdimension);

  std::seed_seq seq{baseSeed, rank}; // Flatten these coordinates, gives a unique number for each processor.
  rng.seed(seq);

  /*seq.generate(seedArr.begin(), seedArr.end());

  ptrdiff_t count = 0;

  for(int z = 0; z < zdimension; ++z){
    rngs.emplace_back(std::vector<std::vector<std::vector<RNGType>>>());
    for(int y = 0; y < ydimension; ++y){
      rngs.back().emplace_back(std::vector<std::vector<RNGType>>());
      for(int x = 0; x < xdimension; ++x){
        rngs.back().back().emplace_back(std::vector<RNGType>());
        for(int l = 0; l<ndof; ++l){
          rngs.back().back().back().emplace_back(RNGType(seedArr[count]));
          count++;
        }
      }
    }
  }*/
}


PetscErrorCode fill( Vec* U, void* ptr)
{
    global_data     *user=(global_data*) ptr;
    model_data      data=user->model;
    PetscInt      i,j,k,l,xstart,ystart,zstart,xdimension,ydimension,zdimension;
    //This function get the information of the global dimension of the grid.
    //DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,&Mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);


    //This Get a pointer to do the calculation
    o4_node ***u;
    DMDAVecGetArray(user->da,*U,&u);

    //Get the Local Corner od the vector
    DMDAGetCorners(user->da,&xstart,&ystart,&zstart,&xdimension,&ydimension,&zdimension);

    //This is the actual computation of the thing
    for (k=zstart; k<zstart+zdimension; k++){
        for (j=ystart; j<ystart+ydimension; j++){
            for (i=xstart; i<xstart+xdimension; i++) {
                for (l=0; l<data.Ndof; l++) {
                  //u[k][j][i].f[l] = normDist(rngs[k-zstart][j-ystart][i-xstart][l]);
                  u[k][j][i].f[l] = normDist(rng);
                }
            }
        }
    }
    DMDAVecRestoreArray(user->da,*U,&u);
    return(0);
}


private:

  //std::vector<std::vector<std::vector<std::vector<RNGType>>>> rngs;
  RNGType rng;

  std::normal_distribution<double> normDist;


};
#endif
