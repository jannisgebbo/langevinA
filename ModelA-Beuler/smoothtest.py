import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

plt.ioff()

def ls(name) :
    print(name)

def printh5():
    with h5py.File('smoothtest_ic.h5','r') as f:
        f.visit(ls)

with h5py.File('smoothtest1_phi.h5','r') as f:
    p1 = f['phi_final'][16,:,:,1]

with h5py.File('smoothtest2_phi.h5','r') as f:
    p2 = f['phi_final'][16,:,:,1]

with h5py.File('smoothtest3_phi.h5','r') as f:
    p3 = f['phi_final'][16,:,:,1]

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3) 

ax1.matshow(p1) ;
ax2.matshow(p2) ;
ax3.matshow(p3) ;
ax4.matshow(p1-p1) ;
ax5.matshow(p2-p1) ;
ax6.matshow(p3-p1) ;
plt.show()  

