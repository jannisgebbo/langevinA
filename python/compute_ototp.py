from measurements import *
import time

if __name__ == '__main__':
    fn = "../data/zminus2_N080_m-0501265_h003684_c00500.h5"
    dt = 0.8
    
    print(fn)
    
    t0 = time.time()

    for k in ["A", "phi", "V", "phi0"]:
    	data = ConfResults(fn = fn,thTime=1000,dt=dt,  data_format="new")
    	data.computeOtOtpBlocked(k, momNum = 3, tMax = 5000.0, blockSizeT = 20000.0,  errFunc = lambda x : bootstrap(x,50), parallel=True)
    
    	data.computeStatisticalCor(k, omMax=0.1, errFunc=lambda x: bootstrap(x,100), filterFunc=lambda x : np.exp(-x / 3000.0))
    	print(time.time() - t0)

    
    	data.save("OtOttp",k)    
    	data.save("OtOttp_blocks",k)    
    	data.save("OtOttpFourier", k)
