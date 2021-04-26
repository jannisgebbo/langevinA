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
   PetscInt Ndof=4;
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
   PetscInt N=model.NX;
};

PetscErrorCode read_o4_data(int argc, char **argv, global_data &data) {
 FCN::ParameterParser par(argc, argv);

 data.model.NX = par.get<int>("NX");
 data.model.NY = par.get<int>("NY");
 data.model.NZ = par.get<int>("NZ");
 data.model.LX = par.get<double>("LX");
 data.model.LY = par.get<double>("LY");
 data.model.LZ = par.get<double>("LZ");
 data.model.Ndof = par.get<int>("Ndof");
 data.model.finaltime = par.get<double>("finaltime");
 data.model.initialtime = par.get<double>("initialtime");
 data.model.deltat = par.get<double>("deltat");
 data.model.mass = par.get<double>("mass");
 data.model.lambda = par.get<double>("lambda");
 data.model.gamma = par.get<double>("gamma");
 data.model.H = par.get<double>("H");
 data.model.seed = par.getSeed("seed");
 data.filename = par.get<std::string>("output");

 PetscReal saveFreqReal = par.get<double>("saveFreq");
 data.model.saveFreq = saveFreqReal / data.model.deltat;


 data.model.hX= data.model.LX/(PetscReal)(data.model.NX-1);
 data.model.hY= data.model.LY/(PetscReal)(data.model.NY-1);
 data.model.hZ= data.model.LZ/(PetscReal)(data.model.NZ-1);


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


 return(0);
}



#endif
