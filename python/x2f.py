from measurements import *
import time
import argparse

#usage:
#
# python3 x2f.py fileToProcess.h5 'X Y Z' 'A1 A2 V1'
#
# to compute the fourier transform of the wall in x y and z of the fields A1 A2 V1.

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('fn', type=str, help='hdf5 filename to process')
    parser.add_argument('dir', type=str, help='directions to process')
    parser.add_argument('flds', type=str, help='fields to process')
    parsed = parser.parse_args()
    
    fn = parsed.fn
    directions = parsed.dir.split(' ')
    fields = parsed.flds.split(' ')
    
    t0 = time.time()
    for direc in directions:
        for fld in fields:
            data = ConfResults(fn = fn,thTime=0,dt=0,  data_format="new")
            data.computeWallFourier(fld, direc)
            data.saveWallFourier(fld, direc)
            
            print(direc + " " + fld + ": {}".format(time.time()  - t0))
            t0 = time.time()
            
