### Compilation on Perlmutter ###
=================================

There is a script which sets up the environment called

setupprlm.sh 

Just do 

source setupprlm.sh

Then the  the code should compile with 

make -f Makefile.prlm 



# 1.  Load the e4s stack

module load e4s

# 2. Switch the spack stack

spack env activate gcc
spack load petsc

# FFTW
# 3. We need to load fftw

module load cray-fftw



