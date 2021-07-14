
mpiexec -n 8 ./noisetest.exe  -ksp_converged_reason -ksp_type cg -pc_type asm -ksp_monitor ascii -snes_view ascii input=noisetest.in

