import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py


f = h5py.File('output.h5','r')  # open the file

solution = f["Timestepsolution/solution"]  #grab the solution 

print(solution)  # printout what it is (some hdf5 type)

timesteps_p0 = solution[20,:,:,8,1]  # read the dataset from the file into a numpy matrix

plt.contourf(timesteps_p0)  # make a contour plot
plt.show()



