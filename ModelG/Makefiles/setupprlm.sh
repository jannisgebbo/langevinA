# Note: spack/0.22 is said to be deprecated and replaced by a newer version but the new one has not 
# got any available environments
module load spack/0.22
spack env activate gcc
spack load petsc
module load cray-fftw
