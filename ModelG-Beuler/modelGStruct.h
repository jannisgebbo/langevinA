#ifndef MODELASTRUCT
#define MODELASTRUCT

#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscviewerhdf5.h>



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
   PetscInt Ndof=10;
   //
   PetscReal finaltime=20.;
   PetscReal initialtime=0.;
   //Time Step
   PetscReal deltat=0.01;
   //Save frequency
   PetscInt saveFreq;


   //Put here the information for the model ex the mass
   //The mass in the action has 1/2 and is the actuall mass square and the quartic has intead 1/4
   PetscReal mass= -5.;
   PetscReal lambda=5.;
   PetscReal gamma=1.;
   PetscReal H=0.;
   PetscReal sigma=1.;
   PetscReal chi=1.;


   // random seed

   PetscInt seed = 10;
   //The name of the file can be put here

};

typedef struct {
 PetscScalar f[10];
} o4_node;

struct global_data  {
   DM da;
   model_data model;
   //Solution
   Vec solution,auxsolution;
   //Previous solution
   Vec previoussolution;
   //Global vector with the stored noise
   Vec noise;
   Vec localnoise;
   //Momentum
   Vec phidot;

   //Jacobian
   Mat jacob;


   // Tag labelling the run. All output files are tag_foo.txt, or tag_bar.h5
   std::string filename = "o4output";

   // Random number generator of gsl
   gsl_rng *rndm;

   // mesuemrnet stuff
   PetscInt N=model.NX;


   global_data(FCN::ParameterParser& par)
   {
     // Here we set up the dimension and the number of fields for now peridic boundary condition.
     read_o4_data(par);
     PetscInt stencil_width =1 ;
     //
     DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,model.NX,
                  model.NY,model.NZ,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, model.Ndof,stencil_width,NULL,NULL,NULL,&da);
    // DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,N, M, PETSC_DECIDE, PETSC_DECIDE, Ndof,  stencil_width, NULL, NULL, &da) ;
     DMSetFromOptions(da) ;
     DMSetUp(da) ;
     //PetscPrintf(PETSC_COMM_SELF," hx = %g, hy = %g\n",(double)model.hX,(double)model.hY);
     //Extract the vector structure
     DMCreateGlobalVector(da,&auxsolution);
     VecDuplicate(auxsolution,&solution);
     VecDuplicate(auxsolution,&noise);
     VecDuplicate(auxsolution,&previoussolution);
     VecDuplicate(auxsolution,&phidot);


     //Set the Jacobian matrix
     DMSetMatType(da,MATAIJ);
     DMCreateMatrix(da,&jacob);

     //Setup the random number generation
     rndm = gsl_rng_alloc(gsl_rng_default);
     gsl_rng_set(rndm, model.seed );
   }

   void finalize()
   {
     gsl_rng_free(rndm);
     MatDestroy(&jacob);
     VecDestroy(&solution);
     VecDestroy(&auxsolution);
     VecDestroy(&noise);
     VecDestroy(&previoussolution);
     VecDestroy(&phidot);
     DMDestroy(&da);
   }

   PetscErrorCode read_o4_data(FCN::ParameterParser& par) {

    model.NX = par.get<int>("NX");
    model.NY = par.get<int>("NY");
    model.NZ = par.get<int>("NZ");
    model.LX = par.get<double>("LX");
    model.LY = par.get<double>("LY");
    model.LZ = par.get<double>("LZ");
    model.Ndof = par.get<int>("Ndof");
    model.finaltime = par.get<double>("finaltime");
    model.initialtime = par.get<double>("initialtime");
    model.deltat = par.get<double>("deltat");
    model.mass = par.get<double>("mass");
    model.lambda = par.get<double>("lambda");
    model.gamma = par.get<double>("gamma");
    model.sigma = par.get<double>("sigma");
    model.chi = par.get<double>("chi");
    model.H = par.get<double>("H");
    model.seed = par.getSeed("seed");
    filename = par.get<std::string>("output");

    PetscReal saveFreqReal = par.get<double>("saveFreq");
    model.saveFreq = saveFreqReal / model.deltat;


    model.hX= model.LX/(PetscReal)(model.NX-1);
    model.hY= model.LY/(PetscReal)(model.NY-1);
    model.hZ= model.LZ/(PetscReal)(model.NZ-1);


    PetscPrintf(PETSC_COMM_WORLD, "NX = %d\n", model.NX);
    PetscPrintf(PETSC_COMM_WORLD, "NY = %d\n", model.NY);
    PetscPrintf(PETSC_COMM_WORLD, "NZ = %d\n", model.NZ);
    PetscPrintf(PETSC_COMM_WORLD, "hX = %e\n", model.hX);
    PetscPrintf(PETSC_COMM_WORLD, "hY = %e\n", model.hY);
    PetscPrintf(PETSC_COMM_WORLD, "hZ = %e\n", model.hZ);
    PetscPrintf(PETSC_COMM_WORLD, "initialtime = %e\n", model.initialtime);
    PetscPrintf(PETSC_COMM_WORLD, "finaltime = %e\n", model.finaltime);
    PetscPrintf(PETSC_COMM_WORLD, "delta t  = %e\n", model.deltat);
    PetscPrintf(PETSC_COMM_WORLD, "seed = %d\n", model.seed);

    PetscPrintf(PETSC_COMM_WORLD, "m2 = %e\n", model.mass);
    PetscPrintf(PETSC_COMM_WORLD, "lambda = %e\n", model.lambda);
    PetscPrintf(PETSC_COMM_WORLD, "sigma = %e\n", model.sigma);
    PetscPrintf(PETSC_COMM_WORLD, "chi = %e\n", model.chi);
    PetscPrintf(PETSC_COMM_WORLD, "H = %e\n", model.H);
    PetscPrintf(PETSC_COMM_WORLD, "filename = %s\n", filename.c_str());


    return(0);
   }
};

o4_node localtimederivative(o4_node *phi, o4_node *phixminus, o4_node *phixplus, o4_node *phiyminus, o4_node *phiyplus, o4_node *phizminus, o4_node *phizplus, void *ptr){
    
    global_data     *user=(global_data*) ptr;
    model_data      data=user->model;
    PetscReal hx=data.hX; //1.0/(PetscReal)(Mx-1);
    PetscReal hy=data.hX; //1.0/(PetscReal)(My-1);
    PetscReal hz=data.hX; //1.0/(PetscReal)(Mz-1);
    o4_node phidot;
    //l is an index for our vector. l=0,1,2,3 is phi,l=4,5,6 is V and l=7,8,9 is A
    //std::cout << "bad Job: " ;
    //computing phi squared
    PetscScalar phisquare=0.;
    for (PetscInt l=0; l<4; l++){
        phisquare  = phisquare+ phi->f[l] * phi->f[l];
    }

    //adotphi vector, to be contracted in the next loop
    PetscScalar vdotphi=0.;
    PetscScalar adotphi[3] = {0., 0., 0};
    //for (l=0; l<3;l++) {
    //  adotphi[l]=0;
    //}

    for (PetscInt l=0; l<3; l++) {
        //contraction of vector current with phi
        vdotphi += phi->f[l+1] *phi->f[l+4];
    }
    
    for(PetscInt l=0; l<3; l++){
         PetscInt   m=(l+1)%3;
         PetscInt   n=(l+2)%3;
         PetscScalar epsilon=((PetscScalar) (l-m)*(m-n)*(n-l))/2.;
        //contraction of axial current with phi
        adotphi[l]=epsilon*(phi->f[m+1]*phi->f[n+7]-phi->f[n+1]*phi->f[m+7]);
    }

    //the phi_zero equation
    PetscScalar ucentral= phi->f[0];
    PetscScalar uxx= ( -2.0* ucentral + phixminus->f[0] + phixplus->f[0] )/(hx*hx);
    PetscScalar uyy= ( -2.0* ucentral + phiyminus->f[0] + phiyplus->f[0] )/(hy*hy);
    PetscScalar uzz= ( -2.0* ucentral + phizminus->f[0] + phizplus->f[0] )/(hz*hz);

     phidot.f[0] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral+  data.gamma*data.H-1/data.chi*vdotphi;
    
   // phidot[k][j][i].f[0] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral+  data.gamma*data.H-1/data.chi*vdotphi+ PetscSqrtReal(2.* data.gamma / data.deltat) * gaussiannoise[k][j][i].f[0];

    //phi_i equation
    for (PetscInt l=1; l<4; l++) {
            ucentral= phi->f[l];
            uxx= ( -2.0* ucentral + phixminus->f[l] + phixplus->f[l] )/(hx*hx);
            uyy= ( -2.0* ucentral + phiyminus->f[l] + phiyplus->f[l] )/(hy*hy);
            uzz= ( -2.0* ucentral + phizminus->f[l] + phizplus->f[l] )/(hz*hz);

        phidot.f[l] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral+1/data.chi*phi->f[0]*phi->f[l+3]-1/data.chi*adotphi[l-1];
        
        //phidot[k][j][i].f[l] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral+1/data.chi*phi[k][j][i].f[0]*phi[k][j][i].f[l+3]-1/data.chi*adotphi[l-1]+  PetscSqrtReal(2.* data.gamma / data.deltat) * gaussiannoise[k][j][i].f[l];
            }

    //v_s equation
    for (PetscInt l=4; l<7; l++) {
            ucentral= phi->f[l];
            uxx= ( -2.0* ucentral + phixminus->f[l] + phixplus->f[l] )/(hx*hx);
            uyy= ( -2.0* ucentral + phiyminus->f[l] + phiyplus->f[l] )/(hy*hy);
            uzz= ( -2.0* ucentral + phizminus->f[l] + phizplus->f[l] )/(hz*hz);

            //advection term
        PetscScalar advectionxx=(-phixplus->f[0]*phi->f[l-3]
                         +phi->f[0]*phixplus->f[l-3]
                         +phi->f[0]*phixminus->f[l-3]
                         -phixminus->f[0]*phi->f[l-3])/(hx*hx);
        PetscScalar advectionyy=(-phiyplus->f[0]*phi->f[l-3]
                         +phi->f[0]*phiyplus->f[l-3]
                         +phi->f[0]*phiyminus->f[l-3]
                         -phiyminus->f[0]*phi->f[l-3])/(hy*hy);
        PetscScalar advectionzz=(-phizplus->f[0]*phi->f[l-3]
                         +phi->f[0]*phizplus->f[l-3]
                         +phi->f[0]*phizminus->f[l-3]
                         -phizminus->f[0]*phi->f[l-3])/(hz*hz);

        phidot.f[l] = data.sigma/data.chi*(uxx+uyy+uzz)+advectionxx+advectionyy+advectionzz -data.gamma*(data.mass+data.lambda*phisquare)*ucentral-data.H*phi->f[l-3];
        
            
        //phidot[k][j][i].f[l] = data.sigma/data.chi*(uxx+uyy+uzz)+advectionxx+advectionyy+advectionzz -data.gamma*(data.mass+data.lambda*phisquare)*ucentral-data.H*phi[k][j][i].f[l-3]+  PetscSqrtReal(2.* data.sigma/data.chi / data.deltat) * gaussiannoise[k][j][i].f[l];
        }

    //a_i equation
    for (PetscInt l=7; l<10; l++) {
            ucentral= phi->f[l];
            uxx= ( -2.0* ucentral + phixminus->f[l] + phixplus->f[l] )/(hx*hx);
            uyy= ( -2.0* ucentral + phiyminus->f[l] + phiyplus->f[l] )/(hy*hy);
            uzz= ( -2.0* ucentral + phizminus->f[l] + phizplus->f[l] )/(hz*hz);

        PetscInt s=l-7;
        PetscInt s1=(s+1)%3;
        PetscInt s2=(s+2)%3;
        PetscScalar epsilon=((PetscScalar) (s-s1)*(s1-s2)*(s2-s))/2.;
            //advection term with epsilon
        PetscScalar advectionxx=epsilon*(phixplus->f[1+s2]*phi->f[1+s1]
                                             +phi->f[1+s2]*phixminus->f[1+s1]
                                             -(phixplus->f[1+s1]*phi->f[1+s2]
                                               +phi->f[1+s1]*phixminus->f[1+s2]))/(hx*hx);
        PetscScalar advectionyy=epsilon*(phiyplus->f[1+s2]*phi->f[1+s1]
                                             +phi->f[1+s2]*phiyminus->f[1+s1]
                                             -(phiyplus->f[1+s1]*phi->f[1+s2]
                                               +phi->f[1+s1]*phiyminus->f[1+s2]))/(hy*hy);
        PetscScalar advectionzz=epsilon*(phizplus->f[1+s2]*phi->f[1+s1]
                                             +phi->f[1+s2]*phizminus->f[1+s1]
                                             -(phizplus->f[1+s1]*phi->f[1+s2]
                                               +phi->f[1+s1]*phizminus->f[1+s2]))/(hz*hz);
        phidot.f[l] = data.sigma/data.chi*(uxx+uyy+uzz)+advectionxx+advectionyy+advectionzz;
        //phidot[k][j][i].f[l] = data.sigma/data.chi*(uxx+uyy+uzz)+advectionxx+advectionyy+advectionzz+PetscSqrtReal(2.* data.sigma/data.chi / data.deltat) * gaussiannoise[k][j][i].f[l];

        }
    return (phidot);
};



#endif
