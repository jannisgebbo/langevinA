#ifndef MODELASTRUCT
#define MODELASTRUCT

#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscviewerhdf5.h>
#include <fstream>




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
   //Save frequency
   PetscInt saveFreq;

   bool verboseMeas;


   //Put here the information for the model ex the mass
   //The mass in the action has 1/2 and is the actuall mass square and the quartic has intead 1/4
   PetscReal mass= -5.;
   PetscReal lambda=5.;
   PetscReal gamma=1.;
   PetscReal H=0.;

   bool zeroStart;

   // random seed

   PetscInt seed = 10;
   //The name of the file can be put here


   std:: string initFile;
   bool coldStart;
    // 1 is BEuler, 2 is FEuler
    PetscInt evolverType=2;
};

typedef struct {
 PetscScalar f[4];
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


   // Viewer
   PetscViewer initViewer;


   // Tag labelling the run. All output files are tag_foo.txt, or tag_bar.h5
   std::string filename = "o4output";


   // Random number generator of gsl
   gsl_rng *rndm;

   // mesuemrnet stuff
   PetscInt N;


   global_data(FCN::ParameterParser& par)
   {
     // Here we set up the dimension and the number of fields for now peridic boundary condition.
     read_o4_data(par);
     N = model.NX;
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

     if(model.coldStart == false) loadFromDereksFile();

     //Setup the random number generation
     rndm = gsl_rng_alloc(gsl_rng_default);
     gsl_rng_set(rndm, model.seed );
   }

   PetscErrorCode loadFromDereksFile(){
     // Initialize a stored initial conditon
     PetscViewerHDF5Open(PETSC_COMM_WORLD, model.initFile.c_str(), FILE_MODE_READ, &initViewer);
     PetscViewerSetFromOptions(initViewer);
     PetscObjectSetName((PetscObject)solution, "final_phi");
     PetscErrorCode ierr = VecLoad(solution, initViewer);
     CHKERRQ(ierr);
     PetscObjectSetName((PetscObject)solution, "o4fields");
     PetscViewerDestroy(&initViewer);
     return ierr;
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
    model.H = par.get<double>("H");
    model.seed = par.getSeed("seed");
    model.verboseMeas = par.get<bool>("verboseMeas",false);
    model.evolverType = par.get<int>("evolverType");

    filename = par.get<std::string>("output");

    model.zeroStart = par.get<bool>("zero_start",false);

    model.initFile = par.get<std::string>("initFile", "x");
    if(model.initFile == "x") model.coldStart = true;
    else model.coldStart = false;

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
    PetscPrintf(PETSC_COMM_WORLD, "H = %e\n", model.H);
    PetscPrintf(PETSC_COMM_WORLD, "filename = %s\n", filename.c_str());

    std::ofstream infofile(filename + ".infos");
    infofile << par;
    infofile.close();

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

    
    for (PetscInt l=0; l<4; l++) {
        PetscScalar ucentral= phi->f[l];
        PetscScalar uxx= ( -2.0* ucentral + phixminus->f[l] + phixplus->f[l] )/(hx*hx);
        PetscScalar uyy= ( -2.0* ucentral + phiyminus->f[l] + phiyplus->f[l] )/(hy*hy);
        PetscScalar uzz= ( -2.0* ucentral + phizminus->f[l] + phizplus->f[l] )/(hz*hz);

        phidot.f[l] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral
        +  ( l==0 ? data.gamma*data.H : 0. );
        
        //phidot[k][j][i].f[l] = data.gamma*(uxx+uyy+uzz)-data.gamma*(data.mass+data.lambda*phisquare)*ucentral+1/data.chi*phi[k][j][i].f[0]*phi[k][j][i].f[l+3]-1/data.chi*adotphi[l-1]+  PetscSqrtReal(2.* data.gamma / data.deltat) * gaussiannoise[k][j][i].f[l];
            }

    return (phidot);
};


PetscErrorCode FormFunctionFEuler( Vec U, void *ptr )
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
    //PetscReal hx=data.hX; //1.0/(PetscReal)(Mx-1);
    //PetscReal hy=data.hX; //1.0/(PetscReal)(My-1);
    //PetscReal hz=data.hX; //1.0/(PetscReal)(Mz-1);


    //take the global vector U and distribute to the local vector localU
    DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);
    DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);



    //From the vector define the pointer for the field u and the rhs f
    o4_node ***phi;//, ***noise;
    DMDAVecGetArray(da,localU,&phi);

    o4_node ***phinew;//, ***noise;
    DMDAVecGetArray(da,U,&phinew);
    
    o4_node ***gaussiannoise;
    DMDAVecGetArrayRead(da,user->noise,&gaussiannoise);

    //o4_node ***oldphi;
    //DMDAVecGetArrayRead(da,user->previoussolution,&oldphi);

    o4_node ***phidot;
    DMDAVecGetArray(da,user->phidot,&phidot);


    //Get The local coordinates
    DMDAGetCorners(da,&xstart,&ystart,&zstart,&xdimension,&ydimension,&zdimension);

    //This actually compute the right hand side
    //PetscScalar uxx,uyy,uzz,ucentral,phisquare;
    for (k=zstart; k<zstart+zdimension; k++){
        for (j=ystart; j<ystart+ydimension; j++){
            for (i=xstart; i<xstart+xdimension; i++) {
                
                o4_node derivative= localtimederivative(&phi[k][j][i], &phi[k][j][i-1], &phi[k][j][i+1], &phi[k][j-1][i], &phi[k][j+1][i], &phi[k-1][j][i], &phi[k+1][j][i],ptr);
                for ( l=0; l<4; l++) {
                    phidot[k][j][i].f[l] =
                    derivative.f[l]
                        +  PetscSqrtReal(2.* data.gamma / data.deltat) * gaussiannoise[k][j][i].f[l];

                        //here you want to put the formula for the euler step F(phi)=0
                        phinew[k][j][i].f[l] +=  data.deltat * phidot[k][j][i].f[l];
                }
            }
        }
    }
    //We need to restore the vector in U and F
    DMDAVecRestoreArray(da,localU,&phi);
    
    
    DMDAVecRestoreArray(da,U,&phinew);
    DMRestoreLocalVector(da,&localU);
    
    
    DMDAVecRestoreArrayRead(da,user->noise,&gaussiannoise);
    //DMDAVecRestoreArrayRead(da,user->previoussolution,&oldphi);

    DMDAVecRestoreArray(da,user->phidot,&phidot);


    return(0);


}


#endif
