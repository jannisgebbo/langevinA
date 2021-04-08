
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
//
// Hello World in petsc
//
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscviewerhdf5.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>


static char help[] = "Make a Grid and tray to solve the heat equation in 3d for 4 field \n\n";

 struct model_data {
    //Lattice dimension
    PetscInt NX = 16;
    PetscInt NY = 16;
    PetscInt NZ = 16;
    //Lattice size
    PetscReal LX= 1.;
    PetscReal LY= 1.;
    PetscReal LZ= 1.;
    
    //Lattice space
    PetscReal hX= LX/(PetscReal)(NX-1);
    PetscReal hY= LY/(PetscReal)(NY-1);
    PetscReal hZ= LZ/(PetscReal)(NZ-1);
    
    //Number of field
    PetscInt Ndof=4;
    //
    PetscReal finaltime=20.;
    PetscReal initialtime=0.;
    //Time Step
    PetscReal deltat=0.01;
    
    
    //Put here the information for the model ex the mass
    //The mass in the action has 1/2 and is the actuall mass square and the quartic has intead 1/4
    PetscReal mass= -5.;
    PetscReal lambda=5.;
    PetscReal gamma=1.;
    PetscReal H=0.;
    
    
    // random seed
    
    PetscInt seed = 10;
    //The name of the file can be put here
    
 };

typedef struct {
  PetscScalar f[4];
} o4_node;

 struct global_data  {
    DM da;
    model_data model;
    // Viewer
    PetscViewer viewer;
    //Previous solution
    Vec previoussolution;
    //Global vector with the stored noise
    Vec noise;
    Vec localnoise;
    //
    Vec phi;
    Vec localphi;
   
    // Tag labelling the run. All output files are tag_foo.txt, or tag_bar.h5
    std::string filename = "o4output";
    
    // Random number generator of gsl
    gsl_rng *rndm;
    
    // mesuemrnet stuff
    PetscInt nobservable=model.Ndof+1;
    Vec Oglobal;
    Vec C11;
    Vec C00;
    PetscInt N=model.NX;
 };

//! Reads the o4_data struct from the command line interface, and priints out
//! the results to stdout. Writes the inputs to a
PetscErrorCode read_o4_data(global_data &data) {
 
    //global_data     *user=(global_data*) ptr;
    //global_data     data=*user;
    
  // The first option is the default global database, with NULL indicating the
  // default.  The second option is the tag of the option set -o4_data_NX
  PetscInt ierr;
  ierr = PetscOptionsGetInt(NULL, "o4_data_", "-NX", &data.model.NX, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, "o4_data_", "-NY", &data.model.NY, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, "o4_data_", "-NZ", &data.model.NZ, NULL);
  CHKERRQ(ierr);
  ierr =
      PetscOptionsGetReal(NULL, "o4_data_", "-LX", &data.model.LX, NULL);
  CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-LY", &data.model.LY, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-LZ", &data.model.LZ, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetInt(NULL, "o4_data_", "-Ndof", &data.model.Ndof, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-finaltime", &data.model.finaltime, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-initialtime", &data.model.initialtime, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-deltat", &data.model.deltat, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-mass", &data.model.mass, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-lambda", &data.model.lambda, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-gamma", &data.model.gamma, NULL);
    CHKERRQ(ierr);
    ierr =
        PetscOptionsGetReal(NULL, "o4_data_", "-H", &data.model.H, NULL);
    CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, "o4_data_", "-seed", &data.model.seed, NULL);
  CHKERRQ(ierr);

    data.model.hX= data.model.LX/(PetscReal)(data.model.NX-1);
    data.model.hY= data.model.LY/(PetscReal)(data.model.NY-1);
    data.model.hZ= data.model.LZ/(PetscReal)(data.model.NZ-1);
    
  char fname[PETSC_MAX_PATH_LEN] = "o4output";
  ierr = PetscOptionsGetString(NULL, "o4_data_", "-filename", fname,
                               sizeof(fname), NULL);
  CHKERRQ(ierr);
  data.filename = fname;

  PetscPrintf(PETSC_COMM_WORLD, "NX = %d\n", data.model.NX);
  PetscPrintf(PETSC_COMM_WORLD, "NY = %d\n", data.model.NY);
  PetscPrintf(PETSC_COMM_WORLD, "NZ = %d\n", data.model.NZ);
  PetscPrintf(PETSC_COMM_WORLD, "hX = %e\n", data.model.hX);
  PetscPrintf(PETSC_COMM_WORLD, "hY = %e\n", data.model.hY);
  PetscPrintf(PETSC_COMM_WORLD, "hZ = %e\n", data.model.hZ);
  PetscPrintf(PETSC_COMM_WORLD, "initialtime = %e\n", data.model.initialtime);
  PetscPrintf(PETSC_COMM_WORLD, "finaltime = %e\n", data.model.finaltime);
  PetscPrintf(PETSC_COMM_WORLD, "delta t  = %e\n", data.model.deltat);
  PetscPrintf(PETSC_COMM_WORLD, "seed = %d\n", data.model.seed);

  PetscPrintf(PETSC_COMM_WORLD, "m2 = %e\n", data.model.mass);
  PetscPrintf(PETSC_COMM_WORLD, "lambda = %e\n", data.model.lambda);
  PetscPrintf(PETSC_COMM_WORLD, "H = %e\n", data.model.H);
  PetscPrintf(PETSC_COMM_WORLD, "filename = %s\n", data.filename.c_str());

  
  //! The purpose of this loop is to write out the inputs in a format that
  //! can be used later for analysis.
    
  return(0);
}

extern PetscErrorCode initialcondition(DM, Vec,void*);
extern PetscErrorCode FormJacobian(SNES ,Vec ,Mat ,Mat ,void* );
extern PetscErrorCode FormFunction(SNES ,Vec ,Vec ,void*);
extern PetscErrorCode Monitor(TS ,PetscInt ,PetscReal ,Vec ,void*);
extern PetscErrorCode noiseGeneration(Vec* , void* ptr);
extern void measure( Vec*, void* );



int main(int argc, char **argv) 
{
    global_data          user,user1; // collection of variabile
    // Initialization
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

    //Read the data form command line
   
    read_o4_data(user);
   
    // If -quit flag, then just stop before we do anything but gather inputs.
    
    // Petsc option processing
    // PetscInt n ;
    // ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL); CHKERRQ(ierr);

    // 3D array descriptor
    DM da ;
    // Here we set up the dimension and the number of fields for now peridic boundary condition.
    PetscInt dimx = user.model.NX;
    PetscInt dimy = user.model.NY;
    PetscInt dimz = user.model.NZ;
    PetscInt Ndof = user.model.Ndof ;
    PetscInt stencil_width =1 ;
    //
    DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,dimx,
                 dimy,dimz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, Ndof,stencil_width,NULL,NULL,NULL,&da);
   // DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,N, M, PETSC_DECIDE, PETSC_DECIDE, Ndof,  stencil_width, NULL, NULL, &da) ;
    DMSetFromOptions(da) ;
    DMSetUp(da) ;
    user.da=da ; //save the vector structure into the global structure meybe is usueful
    //PetscPrintf(PETSC_COMM_SELF," hx = %g, hy = %g\n",(double)user.model.hX,(double)user.model.hY);
    //Extract the vector structure
    Vec solution,auxsolution;
    DMCreateGlobalVector(da,&auxsolution);
    VecDuplicate(auxsolution,&solution);
    VecDuplicate(auxsolution,&user.noise);
    VecDuplicate(auxsolution,&user.previoussolution);
    
    
    //Set the Jacobian matrix
    Mat jacob;
    DMSetMatType(da,MATAIJ);
    DMCreateMatrix(da,&jacob);
    
    //Create the output file
    
    ierr = PetscObjectSetName((PetscObject) solution, "solution");CHKERRQ(ierr);
    
    std::string name(user.filename + ".h5");
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, name.c_str(), FILE_MODE_WRITE,&user.viewer);
    //ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"output.h5",FILE_MODE_WRITE,&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFromOptions(user.viewer);CHKERRQ(ierr);
    //initialize the mesurment vector
    

    
    
    VecCreateSeq(PETSC_COMM_SELF,user.nobservable,&user.Oglobal);
    VecCreateSeq(PETSC_COMM_SELF,user.N,&user.C00);
    VecCreateSeq(PETSC_COMM_SELF,user.N,&user.C11);
    
    PetscObjectSetName((PetscObject) user.Oglobal, "<phi>");
    PetscObjectSetName((PetscObject) user.C00, "C00");
    PetscObjectSetName((PetscObject) user.C11, "C11");
    
    //Setup the random number generation
    user.rndm = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(user.rndm, user.model.seed );
    
    ierr = PetscObjectSetName((PetscObject) user.noise, "noise");CHKERRQ(ierr);
    
    noiseGeneration(&user.noise,&user);
    
    //ierr = PetscViewerHDF5PushGroup(user.viewer, "/noise");CHKERRQ(ierr);
    //ierr = VecView(user.noise,user.viewer);CHKERRQ(ierr);
    //PetscViewerHDF5PopGroup(user.viewer) ;
    
    
    //PetscPrintf(PETSC_COMM_SELF," flat = %g, flat in -1-1 = %g, gaussian in -1-1 = %g\n",(double)gsl_rng_uniform(user.rndm),(double)gsl_ran_flat(user.rndm, -1., 1.),(double)gsl_ran_gaussian(user.rndm, 2.*user.model.gamma));
    //Create the the non linear solver
    SNES           snes;
    // Create
    SNESCreate(PETSC_COMM_WORLD,&snes);
    // Create the equation
    SNESSetFunction(snes,auxsolution,FormFunction,&user);
    // and the jacobian
    SNESSetJacobian(snes,jacob,jacob,FormJacobian,&user);
    SNESSetFromOptions(snes);
    
    ierr = PetscViewerHDF5PushGroup(user.viewer, "/Timestepsolution");CHKERRQ(ierr);
    
    ierr = PetscViewerHDF5SetTimestep(user.viewer, 0);CHKERRQ(ierr);
    
    
    //Now the intial condition
    initialcondition(da,solution,&user);
    //Copy the iniail condition
    VecCopy(solution,user.previoussolution);
    //mesure the intial observable
    measure(&solution, &user);
    //Print the intial condition
    ierr = VecView(solution, user.viewer);CHKERRQ(ierr);
    PetscViewerHDF5IncrementTimestep(user.viewer);
    //PetscPrintf(PETSC_COMM_SELF," 1 = %g\n",user.model.finaltime);
   
    
    PetscInt            steps=1;
    //Thsi is the loop for the time step
    for (PetscReal time =user.model.initialtime; time<user.model.finaltime; time += user.model.deltat) {
        //generate the noise
        noiseGeneration(&user.noise,&user);
        //solve the non linear equation
        SNESSolve(snes,NULL,solution);
        // mesure the solution
        measure(&solution, &user);
        //print the solution
       // ierr = VecView(solution, user.viewer);CHKERRQ(ierr);
        
        
        //increment the time step in the file
        PetscViewerHDF5IncrementTimestep(user.viewer);
        
        
        //print some information to not get boored during the running
        PetscPrintf(PETSC_COMM_SELF,"Timestep %D: step size = %g, time = %g\n",steps,(double) user.model.deltat ,(double)time);
        //copi the solution in the previous time step
        VecCopy(solution,user.previoussolution);
        //advance the step 
        steps ++;
    }
    
    
    // pop up the viewer
    PetscViewerHDF5PopGroup(user.viewer) ;
    
    //ierr = PetscViewerSetFromOptions(user.viewer);CHKERRQ(ierr);
    //ierr = VecView(solution,user.viewer);CHKERRQ(ierr);

    
    //Destroy Everything
    gsl_rng_free(user.rndm);
    MatDestroy(&jacob);
    VecDestroy(&solution);
    VecDestroy(&auxsolution);
    VecDestroy(&user.noise);
    VecDestroy(&user.Oglobal);
    VecDestroy(&user.C00);
    VecDestroy(&user.C11);
    VecDestroy(&user.previoussolution);
    SNESDestroy(&snes);
    DMDestroy(&da);
    PetscViewerDestroy(&user.viewer) ;
    
    
    
    return PetscFinalize();
}
  
//Evaluate the RHS function

PetscErrorCode FormFunction(SNES snes, Vec U, Vec F, void *ptr )
{
    global_data     *user=(global_data*) ptr;
    model_data      data=user->model;
    PetscInt        i,j,k,l,xstart,ystart,zstart,xdimension,ydimension,zdimension;
    DM              da=user->da;
    
    
    //define a local vector
    Vec localU;
    
    DMGetLocalVector(da,&localU);
    
   
    //Get the Global dimension of the Grid
    //DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,&Mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
   // Mx=data.NX;
   // My=data.NX;
   // Mz=data.NX;
    //Define the spaceing
    PetscReal hx=data.hX; //1.0/(PetscReal)(Mx-1);
    PetscReal hy=data.hX; //1.0/(PetscReal)(My-1);
    PetscReal hz=data.hX; //1.0/(PetscReal)(Mz-1);
    
    
    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);
    
    DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);
    
    
    //take the global vector U and distribute to the local vector localU
    //DMGlobalToLocalBegin(da,user->noise,INSERT_VALUES,user->localnoise);
    //DMGlobalToLocalEnd(da,user->noise,INSERT_VALUES,user->localnoise);
    //From the vector define the pointer for the field u and the rhs f
    o4_node ***phi, ***f;//, ***noise;
    DMDAVecGetArrayRead(da,localU,&phi);
    DMDAVecGetArray(da,F,&f);
    
    o4_node ***gaussiannoise;
    DMDAVecGetArrayRead(da,user->noise,&gaussiannoise);
    
    o4_node ***oldphi;
    DMDAVecGetArrayRead(da,user->previoussolution,&oldphi);
    
    //DMDAVecGetArrayRead(da,user->localnoise,&noise);
    
    //Get The local coordinates
    DMDAGetCorners(da,&xstart,&ystart,&zstart,&xdimension,&ydimension,&zdimension);
    
    //This actually compute the right hand side
    PetscScalar uxx,uyy,uzz,ucentral,phisquare;
    for (k=zstart; k<zstart+ydimension; k++){
        for (j=ystart; j<ystart+ydimension; j++){
            for (i=xstart; i<xstart+xdimension; i++) {
                phisquare=0.;
                for (l=0; l<data.Ndof; l++){
                    phisquare  =phisquare+ phi[k][j][i].f[l] * phi[k][j][i].f[l];
                }
                    for ( l=0; l<data.Ndof; l++) {
                        ucentral= phi[k][j][i].f[l];
                        uxx= ( -2.0* ucentral + phi[k][j][i-1].f[l] + phi[k][j][i+1].f[l] )/(hx*hx);
                        uyy= ( -2.0* ucentral + phi[k][j-1][i].f[l] + phi[k][j+1][i].f[l] )/(hy*hy);
                        uzz= ( -2.0* ucentral + phi[k-1][j][i].f[l] + phi[k+1][j][i].f[l] )/(hz*hz);
                        //here you want to put the formula for the euler step F(phi)=0
                        f[k][j][i].f[l]=-phi[k][j][i].f[l]+oldphi[k][j][i].f[l]+data.deltat*(data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral)
                        +  ( l==0 ? data.deltat*data.gamma*data.H : 0. )
                        +PetscSqrtReal(data.deltat * 2.* data.gamma)*gaussiannoise[k][j][i].f[l];
                }
            }
        }
    }
    //We need to restore the vector in U and F
    DMDAVecRestoreArrayRead(da,localU,&phi);
    //DMDAVecRestoreArrayRead(da,user->localnoise,&noise);
    DMDAVecRestoreArray(da,F,&f);
    DMRestoreLocalVector(da,&localU);
    DMDAVecRestoreArrayRead(da,user->noise,&gaussiannoise);
    DMDAVecRestoreArrayRead(da,user->previoussolution,&oldphi);
    //DMLocalToGlobalBegin(user->da,user->localnoise, INSERT_VALUES, user->noise);
    //DMLocalToGlobalEnd(user->da, user->localnoise, INSERT_VALUES, user->noise);
    
    
    
    
    return(0);
    
    
}
// Jacobian !!
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre, void *ptr)
{
    global_data     *user=(global_data*) ptr;
    DM              da=user->da;
    model_data      data=user->model;
    DMDALocalInfo   info;
    PetscInt        i,j,k,l;
    
    
    //Get the local information and store in info
    DMDAGetLocalInfo(da,&info);
    Vec localU;
    DMGetLocalVector(da,&localU);
    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);
    DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);
    
    //From the vector define the pointer for the field phi
    o4_node ***phi;
    DMDAVecGetArrayRead(da,localU,&phi);
    
    //Define the spaceing
    //PetscReal hx= 1.0/(PetscReal)(info.mx-1);
    //PetscReal hy= 1.0/(PetscReal)(info.my-1);
    //PetscReal hz= 1.0/(PetscReal)(info.mz-1);
    PetscReal hx=data.hX; //1.0/(PetscReal)(Mx-1);
    PetscReal hy=data.hY; //1.0/(PetscReal)(My-1);
    PetscReal hz=data.hZ; //1.0/(PetscReal)(Mz-1);
    
    for (k=info.zs; k<info.zs+info.zm; k++){
        for (j=info.ys; j<info.ys+info.ym; j++){
            for (i=info.xs; i<info.xs+info.xm; i++) {
                PetscScalar phisquare=0.;
                for (l=0; l<data.Ndof; l++){
                    phisquare  =phisquare+ phi[k][j][i].f[l] * phi[k][j][i].f[l];
                }
                for (l=0; l<data.Ndof; l++) {
                    //we define the column
                    PetscInt nc=0;
                    MatStencil row, column[10];
                    PetscScalar value[10];
                    //here we insert the position of the row
                    row.i=i; row.j=j; row.k=k; row.c=l;
                    //here we define de position of the non-vansih column for the given row in total there are 7*4 entries and nc is the total number of column per row
                    //x direction
                    column[nc].i=i-1; column[nc].j=j;  column[nc].k=k; column[nc].c=l;    value[nc++]=data.deltat*1.0/(hx*hx);
                    column[nc].i=i+1; column[nc].j=j;  column[nc].k=k; column[nc].c=l;    value[nc++]=data.deltat*1.0/(hx*hx);
                    //y direction
                    column[nc].i=i; column[nc].j=j-1;  column[nc].k=k; column[nc].c=l;    value[nc++]=data.deltat*1.0/(hy*hy);
                    column[nc].i=i; column[nc].j=j+1;  column[nc].k=k; column[nc].c=l;    value[nc++]=data.deltat*1.0/(hy*hy);
                    //z direction
                    column[nc].i=i; column[nc].j=j;  column[nc].k=k-1; column[nc].c=l;    value[nc++]=data.deltat*1.0/(hz*hz);
                    column[nc].i=i; column[nc].j=j;  column[nc].k=k+1;  column[nc].c=l;    value[nc++]=data.deltat*1.0/(hz*hz);
                    //The central element need a loop over the flavour index of the column (is a full matrix in the flavour index )
                    for (PetscInt m=0; m<data.Ndof; m++) {
                        if (m==l) {
                                column[nc].i=i;  column[nc].j=j;   column[nc].k=k; column[nc].c=l;   value[nc++]=-1.+data.deltat*data.gamma*(-2.0/(hy*hy)-2.0/(hx*hx)-2.0/(hz*hz)-data.mass-data.lambda*(phisquare+2.*phi[k][j][i].f[l] * phi[k][j][i].f[l]));
                        } else {
                                column[nc].i=i; column[nc].j=j;  column[nc].k=k; column[nc].c=m;   value[nc++]=data.deltat*data.gamma*(-2.*data.lambda*phi[k][j][i].f[l] * phi[k][j][i].f[m]);
                        }
                    }
                //here we set the matrix
                MatSetValuesStencil(Jpre,1,&row,nc,column,value,INSERT_VALUES );
                }
            }
        }
    }
    MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);
    if (J!=Jpre) {
        MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    }
    DMDAVecRestoreArrayRead(da,localU,&phi);
    DMRestoreLocalVector(da,&localU);
    
    return(0);
    
}


// Intial condition function
PetscErrorCode initialcondition(DM da, Vec U, void* ptr)
{
    global_data     *user=(global_data*) ptr;
    model_data      data=user->model;
    PetscInt      i,j,k,l,xstart,ystart,zstart,xdimension,ydimension,zdimension;
    //This function get the information of the global dimension of the grid.
    //DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,&Mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
    //Compute the lattice spacing
    PetscReal hx=data.hX; //1.0/(PetscReal)(Mx-1);
    PetscReal hy=data.hY; //1.0/(PetscReal)(My-1);
    PetscReal hz=data.hZ; //1.0/(PetscReal)(Mz-1);
    
    //This Get a pointer to do the calculation
    o4_node ***u;
    DMDAVecGetArray(da,U,&u);
    
    //Get the Local Corner od the vector
    DMDAGetCorners(da,&xstart,&ystart,&zstart,&xdimension,&ydimension,&zdimension);
    
    //This is the actual computation of the thing
    for (k=zstart; k<zstart+zdimension; k++){
        PetscReal z=k*hz;
        for (j=ystart; j<ystart+ydimension; j++){
            PetscReal y=j*hy;
            for (i=xstart; i<xstart+xdimension; i++) {
                PetscReal x=i*hx;
                PetscReal r = PetscSqrtReal((x-.5)*(x-.5)+(y-.5)*(y-.5)+(z-.5)*(z-.5));
                for (l=0; l<data.Ndof; l++) {
                    if (r<1.) u[k][j][i].f[l]=gsl_ran_gaussian(user->rndm,1.);//PetscExpReal(-(1+1.*(PetscReal)l)*r*r*r);
                    else u[k][j][i].f[l]=0.0;
                }
                // u[j][i]=PetscSinReal(2.*10.*M_PI*x)*PetscSinReal(5.*M_PI*y);
            }
        }
    }
    DMDAVecRestoreArray(da,U,&u);
    return(0);
    
    
}
    
  //monitorFunction
PetscErrorCode Monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *ptr)
{
    PetscErrorCode ierr;
    global_data     *user=(global_data*) ptr;
    PetscViewer     hdf5=user->viewer;
    PetscReal dt;
    
    TSGetTimeStep(ts,&dt);
    
    ierr = VecView(u, hdf5);CHKERRQ(ierr);
    PetscViewerHDF5IncrementTimestep(hdf5);
    
    noiseGeneration(&user->noise,user);
    PetscPrintf(PETSC_COMM_SELF,"Timestep %D: step size = %g, time = %g\n",steps,(double)dt,(double)time);
    return(0);
}
   


//Noise Genration

PetscErrorCode noiseGeneration( Vec* U, void* ptr)
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
                    u[k][j][i].f[l]=gsl_ran_gaussian(user->rndm, 1. );
                }
                // u[j][i]=PetscSinReal(2.*10.*M_PI*x)*PetscSinReal(5.*M_PI*y);
            }
        }
    }
    DMDAVecRestoreArray(user->da,*U,&u);
    return(0);
    
    
}



void measure( Vec *solution, void *ptr) {
    global_data     *user=(global_data*) ptr;
    model_data      data=user->model;
    DM              da=user->da;
  
    //Get the local information and store in info
    
    Vec localU;
    DMGetLocalVector(da,&localU);
    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,*solution,INSERT_VALUES,localU);
    DMGlobalToLocalEnd(da,*solution,INSERT_VALUES,localU);
    
    //From the vector define the pointer for the field phi
    o4_node ***phi;
    DMDAVecGetArrayRead(da,localU,&phi);
    
    

    // Number of observables
    const size_t nO = 5;
    PetscScalar O[nO] = {}; // Initialize O to zero

    // Correlation function measuremnts
    PetscInt N = data.NX;
    if (data.NX != data.NY || data.NX != data.NZ) {
        PetscPrintf(
                    PETSC_COMM_WORLD,
                    "Nx, Ny, and Nz must be equal for the correlation analysis to work");
        throw("Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }
    // OX0[i] will contain the average of phi0 at wall x=i.
    // And analagous vectors for the wall averages in the y and z directions
    std::vector<PetscScalar> OX0(N, 0.);
    std::vector<PetscScalar> OY0(N, 0.);
    std::vector<PetscScalar> OZ0(N, 0.);
    
    std::vector<PetscScalar> OX1(N, 0.);
    std::vector<PetscScalar> OY1(N, 0.);
    std::vector<PetscScalar> OZ1(N, 0.);
    
    // Get the ranges
    PetscInt ixs, iys, izs, nx, ny, nz;
    DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
    
    for (int k = izs; k < izs + nz; k++) {
        for (int j = iys; j < iys + ny; j++) {
            for (int i = ixs; i < ixs + nx; i++) {
                for (int l = 0; l < 4; l++) {
                    O[l] += phi[k][j][i].f[l];
                }

                OX0[i] += phi[k][j][i].f[0];
                OY0[j] += phi[k][j][i].f[0];
                OZ0[k] += phi[k][j][i].f[0];
                
                OX1[i] += phi[k][j][i].f[1];
                OY1[j] += phi[k][j][i].f[1];
                OZ1[k] += phi[k][j][i].f[1];
            }
        }
    }
    // Retstore the array
    DMDAVecRestoreArray(da, localU, &phi);

    // Reduce each processor measurements O into one.  The observables
    // must be sums
    PetscScalar O_global[nO] = {};
    MPI_Reduce(O, O_global, nO, MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);
    
    // Compute the norm of the vev
    O_global[4] = 0;
    for (int l = 0; l < 4; l++) {
        O_global[4] += O_global[l] * O_global[l];
    }
    O_global[4] = sqrt(O_global[4]);
    
    // For wall to wall correlation function measurements.
    std::vector<PetscScalar> OX0_global(N, 0.);
    std::vector<PetscScalar> OY0_global(N, 0.);
    std::vector<PetscScalar> OZ0_global(N, 0.);
    
    std::vector<PetscScalar> OX1_global(N, 0.);
    std::vector<PetscScalar> OY1_global(N, 0.);
    std::vector<PetscScalar> OZ1_global(N, 0.);
    
    // Bring all the data x data into one
    MPI_Reduce(OX0.data(), OX0_global.data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
    MPI_Reduce(OY0.data(), OY0_global.data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
    MPI_Reduce(OZ0.data(), OZ0_global.data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);

    MPI_Reduce(OX1.data(), OX1_global.data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
    MPI_Reduce(OY1.data(), OY1_global.data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);
    MPI_Reduce(OZ1.data(), OZ1_global.data(), N, MPIU_SCALAR, MPI_SUM, 0,PETSC_COMM_WORLD);

    std::vector<PetscScalar> C00(N, 0.);
    std::vector<PetscScalar> C11(N, 0.);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int jj = (i + j) % N; // Circulate the index, if i=N-1 j=1, jj=0

            C00[j] +=
            (OX0_global[i] * OX0_global[jj] + OY0_global[i] * OY0_global[jj] + OZ0_global[i] * OZ0_global[jj]) /3.;

            C11[j] +=
          (OX1_global[i] * OX1_global[jj] + OY1_global[i] * OY1_global[jj] +OZ1_global[i] * OZ1_global[jj]) /3.;
        }
    }
    //  // Connected correlation function
    //  for (int j=0; j < N; j++) {
    //     C00[j] -= (O_global[0] * O_global[0]) ;
    //  }
    //Let save the voector in the file
    PetscInt indicescorrelation[user->N];
    for (int i=0; i<user->N; i++) {
        indicescorrelation[i]=i;
    }
    PetscInt indicesobservable[nO];
    for (int i=0; i<nO; i++) {
        indicesobservable[i]=i;
    }
    VecSetValues(user->C00,user->N,indicescorrelation,C00.data(),INSERT_VALUES);
    VecSetValues(user->C11,user->N,indicescorrelation,C11.data(),INSERT_VALUES);
    VecSetValues(user->Oglobal,nO,indicesobservable,O_global,INSERT_VALUES);
    

    
    
    VecAssemblyBegin(user->Oglobal);
    VecAssemblyBegin(user->C00);
    VecAssemblyBegin(user->C11);
    
    VecAssemblyEnd(user->Oglobal);
    VecAssemblyEnd(user->C00);
    VecAssemblyEnd(user->C11);
    
    VecView(user->Oglobal, user->viewer);
    VecView(user->C00, user->viewer);
    VecView(user->C11, user->viewer);
    
    
  
}





