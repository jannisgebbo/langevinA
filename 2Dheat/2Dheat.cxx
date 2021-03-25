
#include<cstdio>
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


static char help[] = "Make a Grid and tray to solve the heat equation\n\n";

typedef struct {
    PetscScalar u, dux, duy;
    DM da;
    PetscViewer viewer;
    
} variabiles ;

extern PetscErrorCode initialcondition(DM, Vec,void*);
extern PetscErrorCode RHSJacobian(TS , PetscReal ,Vec , Mat , Mat , void* );
extern PetscErrorCode RHSfunction(TS , PetscReal , Vec , Vec , void* );
extern PetscErrorCode Monitor(TS ,PetscInt ,PetscReal ,Vec u,void*);


int main(int argc, char **argv) 
{
    variabiles          user; // collection of variabile
    PetscInt            steps;
    // Initialization
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

    // Petsc option processing
    // PetscInt n ;
    // ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL); CHKERRQ(ierr);

    // 2D array descriptor
    DM da ;
    // Here we set up the dimension and the number of fields for now peridic boundary condition.
    PetscInt N = 100;
    PetscInt M = 100;
    PetscInt Ndof = 1 ;
    PetscInt stencil_width =1 ;
    //
    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,N, M, PETSC_DECIDE, PETSC_DECIDE, Ndof,  stencil_width, NULL, NULL, &da) ;
    DMSetFromOptions(da) ;
    DMSetUp(da) ;
    user.da=da ; //save the vector structure into the global structure meybe is usueful
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
    
    PetscReal finaltime= 1;
    TSSetMaxTime(ts,finaltime);
    TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
    
    //Now the intial condition
    initialcondition(da,solution,&user);
    PetscReal deltat= .001;
    TSSetTimeStep(ts,deltat);
    
    // Set up every option
    TSSetFromOptions(ts);
    //Create the folder for the solution at intermidiete time step and inizialize the time step for printing
    ierr = PetscViewerHDF5PushGroup(user.viewer, "/Timestepsolution");CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(user.viewer, 0);CHKERRQ(ierr);
    //Solve the system
    TSSolve(ts,solution);
    TSGetSolveTime(ts,&finaltime);
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
    variabiles     *user=(variabiles*) ptr;
    PetscInt      i,j,xstart,ystart,xdimension,ydimension,Mx,My;
    DM            da;
    
    // get the grid from the time stepping
    TSGetDM(ts,&da);
    //define a local vector
    Vec localU;
    DMGetLocalVector(da,&localU);
    
    //Get the Global dimension of the Grid
    DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
    //Define the spaceing
    PetscReal hx= 1.0/(PetscReal)(Mx-1);
    PetscReal hy= 1.0/(PetscReal)(My-1);
    
    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);
    DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);
    
    //From the vector define the pointer for the field u and the rhs f
    PetscScalar **u, **f;
    DMDAVecGetArrayRead(da,localU,&u);
    DMDAVecGetArray(da,F,&f);
    
    //Get The local coordinates
    DMDAGetCorners(da,&xstart,&ystart,NULL,&xdimension,&ydimension,NULL);
    
    //This actually compute the right hand side
    PetscScalar uxx,uyy,ucentral;
    for (j=ystart; j<ystart+ydimension; j++){
        for (i=xstart; i<xstart+xdimension; i++) {
            ucentral= u[j][i];
            uxx= ( -2.0* ucentral + u[j][i-1] + u[j][i+1] )/(hx*hx);
            uyy= ( -2.0* ucentral + u[j-1][i] + u[j+1][i] )/(hy*hy);
            f[j][i]=uxx+uyy;
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
    variabiles     *user=(variabiles*) ptr;
    DM              da;
    DMDALocalInfo   info;
    PetscInt       i,j;
    
    //Get the grid
    TSGetDM(ts,&da);
    //Get the local information and store in info
    DMDAGetLocalInfo(da,&info);
    //Define the spaceing
    PetscReal hx= 1.0/(PetscReal)(info.mx-1);
    PetscReal hy= 1.0/(PetscReal)(info.my-1);
    
    
    for (j=info.ys; j<info.ys+info.ym; j++){
        for (i=info.xs; i<info.xs+info.xm; i++) {
            //we define the column
            PetscInt nc=0;
            MatStencil row, column[5];
            PetscScalar value[5];
            //here we insert the position of the row
            row.i=i; row.j=j;
            //here we define de position of the non-vansih column for the given row in total there are 5 entries and nc
            column[nc].i=i-1; column[nc].j=j;  value[nc++]=1.0/(hx*hx);
            column[nc].i=i+1; column[nc].j=j;  value[nc++]=1.0/(hx*hx);
            column[nc].i=i; column[nc].j=j-1;  value[nc++]=1.0/(hy*hy);
            column[nc].i=i; column[nc].j=j+1;  value[nc++]=1.0/(hy*hy);
            column[nc].i=i; column[nc].j=j;    value[nc++]=-2.0/(hy*hy)-2.0/(hx*hx);
            //here we set the matrix
            MatSetValuesStencil(Jpre,1,&row,nc,column,value,INSERT_VALUES );
        }
    }
    MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);
    if (J!=Jpre) {
        MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    }
    
    return(0);
    
}


// Intial condition function
PetscErrorCode initialcondition(DM da, Vec U, void* ptr)
{
    variabiles     *user=(variabiles*) ptr;
    PetscInt      i,j,xstart,ystart,xdimension,ydimension,Mx,My;
    //This function get the information of the global dimension of the grid.
    DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
    //Compute the lattice spacing
    PetscReal hx= 1.0/(PetscReal)(Mx-1);
    PetscReal hy= 1.0/(PetscReal)(My-1);
    
    //This Get a pointer to do the calculation
    PetscScalar **u;
    DMDAVecGetArray(da,U,&u);
    
    //Get the Local Corner od the vector
    DMDAGetCorners(da,&xstart,&ystart,NULL,&xdimension,&ydimension,NULL);
    
    //This is the actual computation of the thing
    for (j=ystart; j<ystart+ydimension; j++){
        PetscReal y=j*hy;
        for (i=xstart; i<xstart+xdimension; i++) {
            PetscReal x=i*hx;
            PetscReal r = PetscSqrtReal((x-.5)*(x-.5)+(y-.5)*(y-.5));
            if (r<0.125) u[j][i]=PetscExpReal(-20.*r*r);
            else u[j][i]=0.0;
           // u[j][i]=PetscSinReal(2.*10.*M_PI*x)*PetscSinReal(5.*M_PI*y);
        }
    }
    
    DMDAVecRestoreArray(da,U,&u);
    return(0);
    
    
}
    
  //monitorFunction
PetscErrorCode Monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *ptr)
{
    PetscErrorCode ierr;
    variabiles     *user=(variabiles*) ptr;
    PetscViewer    hdf5=user->viewer;
    PetscReal dt;
    TSGetTimeStep(ts,&dt);
    
    ierr = VecView(u, hdf5);CHKERRQ(ierr);
    PetscViewerHDF5IncrementTimestep(hdf5);
    
    PetscPrintf(PETSC_COMM_SELF,"Timestep %D: step size = %g, time = %g\n",steps,(double)dt,(double)time);
    return(0);
}
   


