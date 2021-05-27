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





#endif
