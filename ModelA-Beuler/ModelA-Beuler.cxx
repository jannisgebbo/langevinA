
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>


//
// Hello World in petsc
//

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>


//Homemade parser
#include "parameterparser/parameterparser.h"

//Measurer, where the Petsc are included

#include "measurer.h"

extern PetscErrorCode initialcondition(DM, Vec,void*);
extern PetscErrorCode FormJacobian(SNES ,Vec ,Mat ,Mat ,void* );
extern PetscErrorCode FormFunction(SNES ,Vec ,Vec ,void*);
extern PetscErrorCode noiseGeneration(Vec* , void* ptr);



int main(int argc, char **argv)
{
    // Initialization
    PetscErrorCode ierr;
    std::string help = "Model A. Call ./ModelA-Beuler.exe input=input.in with input.in your input file.";
    ierr = PetscInitialize(&argc,&argv,(char*)0,help.c_str()); if (ierr) return ierr;

    FCN::ParameterParser par(argc, argv);
    //Read the data form command line
    global_data          user(par); // collection of variable

    //initialize the measurments
    Measurer measurer(&user);


    ierr = PetscObjectSetName((PetscObject) user.noise, "noise");CHKERRQ(ierr);

    noiseGeneration(&user.noise,&user);


    //Create the the non linear solver
    SNES           snes;
    // Create
    SNESCreate(PETSC_COMM_WORLD,&snes);
    // Create the equation
    SNESSetFunction(snes,user.auxsolution,FormFunction,&user);
    // and the jacobian
    SNESSetJacobian(snes,user.jacob,user.jacob,FormJacobian,&user);
    SNESSetFromOptions(snes);


    //Now the intial condition
    initialcondition(user.da,user.solution,&user);
    //Copy the iniail condition
    VecCopy(user.solution,user.previoussolution);
    //mesure the intial observable
    measurer.measure(&user.solution,&user.phidot);

    PetscInt            steps=1;
    //Thsi is the loop for the time step
    for (PetscReal time =user.model.initialtime; time<user.model.finaltime; time += user.model.deltat) {
        //generate the noise
        noiseGeneration(&user.noise,&user);
        //solve the non linear equation
        SNESSolve(snes,NULL,user.solution);
        // mesure the solution
        if(steps % user.model.saveFreq == 0)  measurer.measure(&user.solution,&user.phidot);

        //print some information to not get bored during the running
        PetscPrintf(PETSC_COMM_WORLD,"Timestep %D: step size = %g, time = %g\n",steps,(double) user.model.deltat ,(double)time);
        //copi the solution in the previous time step
        VecCopy(user.solution,user.previoussolution);
        //advance the step
        steps ++;
    }

    //Destroy Everything

    user.finalize();
    SNESDestroy(&snes);
    measurer.finalize();

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
    //Define the spaceing
    PetscReal hx=data.hX; //1.0/(PetscReal)(Mx-1);
    PetscReal hy=data.hX; //1.0/(PetscReal)(My-1);
    PetscReal hz=data.hX; //1.0/(PetscReal)(Mz-1);


    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);
    DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);



    //From the vector define the pointer for the field u and the rhs f
    o4_node ***phi, ***f;//, ***noise;
    DMDAVecGetArrayRead(da,localU,&phi);
    DMDAVecGetArray(da,F,&f);

    o4_node ***gaussiannoise;
    DMDAVecGetArrayRead(da,user->noise,&gaussiannoise);

    o4_node ***oldphi;
    DMDAVecGetArrayRead(da,user->previoussolution,&oldphi);

    o4_node ***phidot;
    DMDAVecGetArray(da,user->phidot,&phidot);


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


                        phidot[k][j][i].f[l] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral
                          +  ( l==0 ? data.gamma*data.H : 0. )
                          +  PetscSqrtReal(2.* data.gamma / data.deltat) * gaussiannoise[k][j][i].f[l];


                        //here you want to put the formula for the euler step F(phi)=0
                        f[k][j][i].f[l]=-phi[k][j][i].f[l] + oldphi[k][j][i].f[l] + data.deltat * phidot[k][j][i].f[l];
                }
            }
        }
    }
    //We need to restore the vector in U and F
    DMDAVecRestoreArrayRead(da,localU,&phi);

    DMDAVecRestoreArray(da,F,&f);
    DMRestoreLocalVector(da,&localU);
    DMDAVecRestoreArrayRead(da,user->noise,&gaussiannoise);
    DMDAVecRestoreArrayRead(da,user->previoussolution,&oldphi);

    DMDAVecRestoreArray(da,user->phidot,&phidot);


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

    //Define the spacing

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
            }
        }
    }
    DMDAVecRestoreArray(user->da,*U,&u);
    return(0);
}
