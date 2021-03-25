
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
    
    //
    PetscReal finaltime=1.;
    PetscReal initialtime=0.;
    //Time Step
    PetscReal deltat=0.05;
    
    
    //Put here the information for the model ex the mass
    
    PetscReal mass=1.;
    PetscReal lambda=0.;
    PetscReal gamma=1.;
    
    //The name of the file can be put here
    
} ;

typedef struct {
  PetscScalar f[4];
} o4_node;

struct global_data {
    DM da;
    model_data model;
    PetscViewer viewer;
    
}  ;

extern PetscErrorCode initialcondition(DM, Vec,void*);
extern PetscErrorCode RHSJacobian(TS , PetscReal ,Vec , Mat , Mat , void* );
extern PetscErrorCode RHSfunction(TS , PetscReal , Vec , Vec , void* );
extern PetscErrorCode Monitor(TS ,PetscInt ,PetscReal ,Vec u,void*);


int main(int argc, char **argv) 
{
    global_data          user; // collection of variabile
    PetscInt            steps;
    // Initialization
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

    // Petsc option processing
    // PetscInt n ;
    // ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL); CHKERRQ(ierr);

    // 3D array descriptor
    DM da ;
    // Here we set up the dimension and the number of fields for now peridic boundary condition.
    PetscInt dimx = user.model.NX;
    PetscInt dimy = user.model.NY;
    PetscInt dimz = user.model.NZ;
    PetscInt Ndof = 4 ;
    PetscInt stencil_width =1 ;
    //
    DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,dimx,
                 dimy,dimz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, Ndof,stencil_width,NULL,NULL,NULL,&da);
   // DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,N, M, PETSC_DECIDE, PETSC_DECIDE, Ndof,  stencil_width, NULL, NULL, &da) ;
    DMSetFromOptions(da) ;
    DMSetUp(da) ;
    user.da=da ; //save the vector structure into the global structure meybe is usueful
    PetscPrintf(PETSC_COMM_SELF," hx = %g, hy = %g\n",(double)user.model.hX,(double)user.model.hY);
    //Extract the vector structure
    Vec solution,auxsolution;
    DMCreateGlobalVector(da,&auxsolution);
    VecDuplicate(auxsolution,&solution);
    
    //Create the output file
    
    ierr = PetscObjectSetName((PetscObject) solution, "solution");CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"output.h5",FILE_MODE_WRITE,&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFromOptions(user.viewer);CHKERRQ(ierr);
    
    
    //Create the time stepping TS
    //Variable definition
    TS ts;
    // Actual cration of the time step
    TSCreate(PETSC_COMM_WORLD,&ts);
    // Set the dm
    TSSetDM(ts,da);
    // properies of the TS
    TSSetProblemType(ts,TS_NONLINEAR);
    TSSetType(ts,TSBEULER);
    //Set the right hand side function called RHSfunction
    TSSetRHSFunction(ts,auxsolution,RHSfunction,&user);
    
    //Monitor
    TSMonitorSet(ts,Monitor,&user,NULL);
    
    //Set the Jacobian matrix
    Mat jacob;
    DMSetMatType(da,MATAIJ);
    DMCreateMatrix(da,&jacob);
    //the two Jac are the approxiamte Jacobian and the precoditionig matrix in this case are the same and are the exact Jacobian.
    TSSetRHSJacobian(ts,jacob,jacob,RHSJacobian,&user);
    
    
    TSSetMaxTime(ts,user.model.finaltime);
    TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
    
    //Now the intial condition
    initialcondition(da,solution,&user);
    
    TSSetTimeStep(ts,user.model.deltat);
    
    // Set up every option
    TSSetFromOptions(ts);
    //Create the folder for the solution at intermidiete time step and inizialize the time step for printing
    ierr = PetscViewerHDF5PushGroup(user.viewer, "/Timestepsolution");CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(user.viewer, 0);CHKERRQ(ierr);
    //Solve the system
    TSSolve(ts,solution);
    TSGetSolveTime(ts,&user.model.finaltime);
    TSGetStepNumber(ts,&steps);
    
    
    // pop up the viewer
    PetscViewerHDF5PopGroup(user.viewer) ;
    
    //ierr = PetscViewerSetFromOptions(user.viewer);CHKERRQ(ierr);
    //ierr = VecView(solution,user.viewer);CHKERRQ(ierr);

    
    //Destroy Everything
    MatDestroy(&jacob);
    VecDestroy(&solution);
    VecDestroy(&auxsolution);
    TSDestroy(&ts);
    DMDestroy(&da);
    PetscViewerDestroy(&user.viewer) ;
    
    
    
    return PetscFinalize();
}
  
//Evaluate the RHS function

PetscErrorCode RHSfunction(TS ts, PetscReal ftime, Vec U, Vec F, void *ptr )
{
    global_data     *user=(global_data*) ptr;
    model_data      data=user->model;
    PetscInt      i,j,k,l,xstart,ystart,zstart,xdimension,ydimension,zdimension;
    DM            da;
    
    // get the grid from the time stepping
    TSGetDM(ts,&da);
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
    
    //From the vector define the pointer for the field u and the rhs f
    o4_node ***u, ***f;
    DMDAVecGetArrayRead(da,localU,&u);
    DMDAVecGetArray(da,F,&f);
    
    //Get The local coordinates
    DMDAGetCorners(da,&xstart,&ystart,&zstart,&xdimension,&ydimension,&zdimension);
    
    //This actually compute the right hand side
    PetscScalar uxx,uyy,uzz,ucentral,phisquare;
    for (k=zstart; k<zstart+ydimension; k++){
        for (j=ystart; j<ystart+ydimension; j++){
            for (i=xstart; i<xstart+xdimension; i++) {
                phisquare=0.;
                for (l=0; l<4; l++){
                    phisquare  =phisquare+ u[k][j][i].f[l] * u[k][j][i].f[l];
                }
                    for ( l=0; l<4; l++) {
                        ucentral= u[k][j][i].f[l];
                        uxx= ( -2.0* ucentral + u[k][j][i-1].f[l] + u[k][j][i+1].f[l] )/(hx*hx);
                        uyy= ( -2.0* ucentral + u[k][j-1][i].f[l] + u[k][j+1][i].f[l] )/(hy*hy);
                        uzz= ( -2.0* ucentral + u[k-1][j][i].f[l] + u[k+1][j][i].f[l] )/(hz*hz);
                        f[k][j][i].f[l]=uxx+uyy+uzz ;
                }
            }
        }
    }
    //We need to restore the vector in U and F
    DMDAVecRestoreArrayRead(da,localU,&u);
    DMDAVecRestoreArray(da,F,&f);
    DMRestoreLocalVector(da,&localU);
    
    
    return(0);
    
    
}
// Jacobian !!
PetscErrorCode RHSJacobian(TS ts, PetscReal t,Vec U, Mat J, Mat Jpre, void *ptr)
{
    global_data     *user=(global_data*) ptr;
    DM              da;
    model_data      data=user->model;
    DMDALocalInfo   info;
    PetscInt        i,j,k,l;
    
    //Get the grid
    TSGetDM(ts,&da);
    //Get the local information and store in info
    DMDAGetLocalInfo(da,&info);
    Vec localU;
    DMGetLocalVector(da,&localU);
    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);
    DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);
    
    //From the vector define the pointer for the field phi
    o4_node ***phi;
    DMDAVecGetArrayRead(da,localU,&u);
    
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
                for (l=0; l<4; l++){
                    phisquare  =phisquare+ phi[k][j][i].f[l] * phi[k][j][i].f[l];
                }
                for (l=0; l<4; l++) {
                //we define the column
                PetscInt nc=0;
                MatStencil row, column[7*4];
                PetscScalar value[7*4];
                //here we insert the position of the row
                    row.i=i; row.j=j; row.k=k; row.c=l;
                //here we define de position of the non-vansih column for the given row in total there are 7*4 entries and nc is the total number of column per row
                //x direction
                    column[nc].i=i-1; column[nc].j=j;  column[nc].k=k; column[nc].c=l;    value[nc++]=1.0/(hx*hx);
                    column[nc].i=i+1; column[nc].j=j;  column[nc].k=k; column[nc].c=l;  value[nc++]=1.0/(hx*hx);
                //y direction
                    column[nc].i=i; column[nc].j=j-1;  column[nc].k=k; column[nc].c=l; value[nc++]=1.0/(hy*hy);
                    column[nc].i=i; column[nc].j=j+1;  column[nc].k=k; column[nc].c=l; value[nc++]=1.0/(hy*hy);
                //z direction
                    column[nc].i=i; column[nc].j=j;  column[nc].k=k-1; column[nc].c=l; value[nc++]=1.0/(hz*hz);
                    column[nc].i=i; column[nc].j=j;  column[nc].k=k+1; column[nc].c=l; value[nc++]=1.0/(hz*hz);
                //The central element
                    column[nc].i=i; column[nc].j=j;  column[nc].k=k; column[nc].c=l;   value[nc++]=-2.0/(hy*hy)-2.0/(hx*hx)-2.0/(hz*hz);
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
    DMDAVecRestoreArrayRead(da,localU,&u);
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
                for (l=0; l<4; l++) {
                    if (r<0.125) u[k][j][i].f[l]=PetscExpReal(-(1+10.*(PetscReal)l)*r*r);
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
    
    PetscPrintf(PETSC_COMM_SELF,"Timestep %D: step size = %g, time = %g\n",steps,(double)dt,(double)time);
    return(0);
}
   


