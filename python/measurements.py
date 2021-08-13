import random
import h5py
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt


class StatResult:
    def __init__(self, tup):
        self.mean = tup[0]
        self.err = tup[1]

#Functions
def bootstrap(arr, nSamples=100):
    arr = np.asarray(arr)
    random.seed()
    bootstrapArr = []
    for n in range(nSamples):
        newArr = np.zeros(np.shape(arr), dtype=arr.dtype)
        for l in range(len(arr)):
            newArr[l] = arr[random.randrange(0,len(arr))]
        bootstrapArr.append(np.mean(newArr,axis = 0))

    bootstrapArr = np.asarray(bootstrapArr)

    return (np.mean(bootstrapArr,axis = 0), np.std(bootstrapArr,axis = 0))
    #return (np.mean(arr,axis = 0), np.std(arr,axis = 0))

def jackknife(arr, nBlock=10):
    nSamples = int(len(arr) / nBlock)
    arr = np.asarray(arr)
    random.seed()
    bootstrapArr = []
    count = 0
    thetaBar = np.mean(arr, axis=0)
    sigma2 = np.zeros(1 if len(np.shape(arr)) == 1 else np.shape(arr[0,:]), dtype=arr.dtype)
    for n in range(nSamples):
        newArr = np.concatenate((arr[:count], arr[count+nBlock:]))
        bootstrapArr.append(np.mean(newArr,axis = 0))
        count+=nBlock
        sigma2 += (bootstrapArr[-1] - thetaBar)**2

    bootstrapArr = np.asarray(bootstrapArr)
    

    
    return (np.mean(bootstrapArr,axis = 0), np.sqrt((nSamples -1.0)/nSamples * sigma2))
    #return (np.mean(arr,axis = 0), np.std(arr,axis = 0))

def autocorrelationFunction(arr):
    mean2 = np.mean(arr,axis = 0)**2
    X2 = np.mean(arr**2,axis = 0)
    newArr = np.zeros(np.shape(arr), dtype=arr.dtype)
    N = len(arr)
    for t in range(N):
        for i in range(N-t):
            newArr[t] += arr[i] * arr[i+t]
        newArr[t]/=float(N-t)
        newArr[t] = (newArr[t] - mean2) / (X2 - mean2)
    print(X2)
    return  newArr
        
#def intAutocorrrelationTime(arr):
#    Ct = autocorrelationFunction(arr)
#    res = 0.5
#    count = 0
    #while count < 5 * res:
      #  res+=

# Compute the correlator in time between arr1 and arr2. Can also substract the connected part.
def computeOtOtp(arr1, arr2, statFunc, conn=False, nTMax=-1, decim=1):
    Npoints = len(arr1)
    if nTMax==-1:
        nTMax=Npoints
        
    
    Cttp = np.zeros(nTMax, dtype=type(arr1[0]))
    CttpErr = np.zeros(nTMax, dtype=type(arr1[0]))
    
    av1 = np.mean(arr1)
    av2 = np.mean(arr2)
    # Points over which we do the time average

    for tt in range(nTMax): #tt is the time difference
        counter = 0
        nStarts = list(range(0, Npoints - tt, decim))
        tmpArr = np.zeros(len(nStarts), dtype=type(arr1[0]))
        
        for t0 in nStarts: # t0 is the origin
            tmpArr[counter] = arr1[t0 + tt] * arr2[t0]
            if conn:
                tmpArr[counter] -= av1 * av2
            counter += 1
        Cttp[tt], CttpErr[tt] = statFunc(tmpArr)
    
    return Cttp, CttpErr



# Compute the O(t) O(0) correlator over blocks in time.
def computeBlockedOtOtp(arr1, arr2, nTMax, steps, decim=1, conn=False):
    res = []
    sMax = len(arr1) - steps
    for s in range(0,sMax,steps):
        tmpR, tmpE = computeOtOtp(arr1[s:s+steps],arr2[s:s+steps],lambda x : (np.mean(x),0) ,nTMax=nTMax,decim=decim,conn=conn)
        res.append(tmpR)
    return np.asarray(res)


# "Slow" fourier transform in time. Used only for the one in time. Can apply a filter
def ft(om, dt, arr2, filterFunc = lambda x : 1):
    tmp = 0.0
    for i in range(len(arr2)):
        t = i * dt
        tmp += arr2[i] * np.exp(1j * om * t) * filterFunc(t)
    return tmp

# transform a whole array to fourier space, with optimal sampling and up to oMax
def toFourier(arr, dt, omMax, filterFunc = lambda x : 1):
    tMax= len(arr)*dt
    omegaIR = 2.0 * np.pi / tMax
    oms = np.arange(0, omMax, omegaIR)
    res = []
    for om in oms:
        res.append(ft(om,dt,arr, filterFunc))

    return (oms, np.asarray(res))

#assume first component is at zero and we don't want to double it.
def symmetrizeArray(arr, sym = 1):
    res = np.zeros(2 * len(arr) - 1 , arr.dtype)
    for i in range(1,len(arr) ):
        res[i-1] = sym * arr[-i]
    for i in range(len(arr)-1,len(res)):
        res[i] = arr[i - len(arr) + 1]
    return res

#Compute the Fourier transform over blocks of data.
def toFourierBlocked(arr, dt, omMax, errFunc, filterFunc = lambda x : 1):
    length = len(arr)
    res = []
    for i in range(length):
        oms, tmp = toFourier(arr[i], dt, omMax, filterFunc)
        res.append(tmp)
    f, ferr = errFunc(np.asarray(res))
    
    
    return (symmetrizeArray(oms,-1), StatResult((symmetrizeArray(f),symmetrizeArray(ferr))))




# For testing purpose only
def manualFourier(arr):
    Nx = len(arr)
    ks = [2 * np.pi * i /float(Nx) for i in range(Nx)]
    arrF = np.zeros(Nx, dtype=complex)
    for k in range(Nx):
        for x in range(Nx):
            arrF[k] += arr[x] * np.exp(-1j * x * ks[k] )

    return arrF




# static correlator in x

def twoPtAv(arr):
    N=len(arr)
    res = np.zeros(N)
    for i in range(N):
        res[i]=0
        for j in range(N):
            res[i]+=arr[j]*arr[(i+j)%N]

    return np.asarray(res)

#classes


#Work in proigress, store the results and call the corersponding measurements in a user friendly way.
class ConfResults:
    def __init__(self,fn, thTime , dt, data_format ="new"):
        self.fn = fn
        self.thTime = thTime
        self.dt = dt
        
        self.data_format = data_format
        
        self.wkeys = dict()
        
        if self.data_format == "old":
            self.wkeys["phi0"] = "wallX_phi_0"
            self.wkeys["dsigma"] = "wallX_phi_0"
            self.wkeys["phi1"] = "wallX_phi_1"
            self.wkeys["phi2"] = "wallX_phi_2"
            self.wkeys["phi3"] = "wallX_phi_3"
            self.wkeys["A1"] = "wallX_phi_4"
            self.wkeys["A2"] = "wallX_phi_5"
            self.wkeys["A3"] = "wallX_phi_6"
            self.wkeys["V1"] = "wallX_phi_7"
            self.wkeys["V2"] = "wallX_phi_8"
            self.wkeys["V3"] = "wallX_phi_9"
        else:
            self.wkeys["phi0"] = 0
            self.wkeys["dsigma"] = 0
            self.wkeys["phi1"] = 1
            self.wkeys["phi2"] = 2
            self.wkeys["phi3"] = 3
            self.wkeys["A1"] = 4
            self.wkeys["A2"] = 5
            self.wkeys["A3"] = 6
            self.wkeys["V1"] = 7
            self.wkeys["V2"] = 8
            self.wkeys["V3"] = 9
        
        #The results are stored in dictionnary, to have a generic way to code between the different fields.
        
        if data_format == "old" or data_format == "semi_old" :
            self.directions = ["X"]
        else :
            self.directions = ["X","Y","Z"]
        
        self.wall = dict()
        self.wallF = dict()
        self.wallCorr = dict()
        self.wallFProp = dict()
        for d in self.directions:
            self.wall[d] = dict()
            self.wallF[d] = dict()
            self.wallCorr[d] = dict() 
        
        self.OtOttp_blocks = dict()
        
        
        self.OtOttp = dict()
        self.OtOttp_time = dict()
        
        #self.OtOttpFourier_blocks = dict()
        self.OtOttpFourier = dict()
        self.OtOttpFourier_oms = dict()
        
        #self.momenta_3d we also define the momenta 3d below
        
    def loadWall(self,key, direc):
        r = h5py.File(self.fn,'r')
        try :
            if not (direc == "X" or  (self.data_format != "old" and self.data_format != "semiold")) :
                raise
            else:
                if self.data_format == "old":
                    self.wall[direc][key] = np.asarray(r[self.wkeys[key]])[self.thTime:] 
                elif self.data_format == "semiold":
                    self.wall[direc][key] = np.asarray(r["corrx"])[self.thTime:,self.wkeys[key],:] 
                else:
                    self.wall[direc][key] = np.asarray(r["wall{}".format(direc.lower())])[self.thTime:,self.wkeys[key],:]
                if key == "dsigma":
                    self.wall[direc][key] -= np.mean(np.mean(self.wall[direc][key], axis = 0))
                
        except:
            print("You tried to load a wall in a direction that does not exist in your data format.")
            

    #TODO: Change the way we handle the average.
    def readAv(self):
        r = h5py.File(self.fn,'r')
        self.phi = np.asarray(r["phi"])[self.thTime:]
        self.phiNorm = self.phi[:,4]

    def computeMag(self, direc = 0):
        self.readAv()
        self.mag, self.magErr = bootstrap(self.phi[:,direc],100)


    #Compute the fourier transform of a given wall, specified by the appropriate key.
    def computeWallFourier(self, key, direc, decim = 1):
        self.loadWall(key, direc)
        Nx = len(self.wall[direc][key][0])
        ts = range(0,len(self.wall[direc][key]), decim)
        self.wallF[direc][key] = np.zeros([len(ts), len(self.wall[direc][key][0,:])], dtype=complex)
        c = 0
        for t in ts:
            self.wallF[direc][key][c] = np.fft.fft(self.wall[direc][key][t]) / Nx
            c += 1
        self.momenta_3d = np.asarray([ 2.0 * np.pi / float(Nx) * n for n in range(Nx)])
            
    #Computes the time correlator of a given key.
    #TODO: For now, compute only from one direction in fourier for all modes. Need to include other direction for non-zero k for better stat.
    def computeOtOtpBlocked(self, key, tMax, blockSizeT, errFunc, momNum = 0, conn = False):
        if not key in self.OtOttp_blocks.keys():
            if key != "phi0":
                keys = [key + str(i) for i in [1,2,3]]
                for k in keys:
                    self.computeWallFourier(k,"X")
                av = np.zeros(np.shape(self.wallF["X"][keys[0]][:,momNum]), dtype = self.wallF["X"][keys[0]][:,momNum].dtype)
                for k in keys:
                    av += self.wallF["X"][k][:,momNum]
                av /= 3.0
            else:
                self.computeWallFourier(key,"X")
                av = self.wallF["X"][key][:,momNum]


            nTMax = int(np.floor(tMax / self.dt))
            blockSize = int(np.floor(blockSizeT / self.dt))

            self.OtOttp_blocks[key] = computeBlockedOtOtp(av,av,nTMax, blockSize, decim = 1, conn = conn)
       
        self.OtOttp[key] = StatResult(errFunc(self.OtOttp_blocks[key]))
        
        self.OtOttp_time[key] = np.arange(0,tMax, self.dt)

    #Compute the statistical correlator of a given key.
    def computeStatisticalCor(self, key, omMax, errFunc, filterFunc = lambda x : 1):
        try :
            print(key in self.OtOttp_blocks.keys())
            if not (key in self.OtOttp_blocks.keys()):
                raise ;
            else:
                self.OtOttpFourier_oms[key],self.OtOttpFourier[key] = toFourierBlocked(self.OtOttp_blocks[key], self.dt, omMax, errFunc, filterFunc)
        except:
            print("the two point function in time of {} was not computed. Need to call computeOtOtpBlocked first.".format(key))
            
            
    def computeFourierPropagator(self, key, decim, errFunc):
        if key != "phi0" and key != "dsigma":
            keys = [key + str(i) for i in [1,2,3]]
            for k in keys:
                for d in self.directions:
                    self.computeWallFourier(k, d, decim)
            tmp = np.zeros(np.shape(self.wallF["X"][keys[0]]), dtype = self.wallF["X"][keys[0]].dtype)
            for k in keys:
                for d in self.directions:
                    tmp += self.wallF[d][k] * np.conj(self.wallF[d][k])
            tmp /= 3.0 / float(len(self.directions))
        else:
            for d in self.directions:
                    self.computeWallFourier(key, d, decim)
            tmp = np.zeros(np.shape(self.wallF["X"][key]), dtype = self.wallF["X"][key].dtype)
            for d in self.directions:
                tmp += self.wallF[d][key] * np.conj(self.wallF[d][key])
            tmp /= 3.0 / float(len(self.directions))

        self.wallFProp[key] = StatResult(errFunc(tmp))

        
class Plotter:
    def __init__(self):
        return

    def plot(self, confRes, obs, key, xfact = 1.0, yfact = 1.0):
        if obs == "OtOttp":
            self.errorplot(xfact * confRes.OtOttp_time[key], confRes.OtOttp[key], yfact = yfact)
        elif obs == "OtOttpFourier":
            self.errorplot(xfact * confRes.OtOttpFourier_oms[key], confRes.OtOttpFourier[key], yfact = yfact)
        elif obs == "propagator":
            self.errorplot(xfact * confRes.momenta_3d, confRes.wallFProp[key], yfact = yfact)



    def errorplot(self, x, statobj, yfact):
        plt.errorbar(x, yfact * statobj.mean, yfact * statobj.err)

''' Deprecated, keep for the idea. Would be better o use the block structure above though.
class EnsembleResults:
    def __init__(self,fn, nEnd, nStart=0):
        self.fnList = [fn +"_"+str(i)+".h5" for i in range(nStart,nEnd+1)]
        self.nConf = nEnd-nStart+1


    def computeWallXFourier0(self, nTherm, nBoot, decim):
        resTot = []
        for fn in self.fnList:
            res = ConfResults(fn, nTherm,decim)
            print(fn, end="\r")
            res.computeWallXFourier0()
            resTot.append(res.wallXphi0F)
        self.wallXFourier0, self.wallXFourier0Err  = bootstrap(resTot, nBoot)
    def computeWallXFourierSquare0(self, nTherm, nBoot, decim):
        resTot = []
        for fn in self.fnList:
            res = ConfResults(fn, nTherm,decim)
            print(fn, end="\r")
            res.computeWallXFourier0()
            resTot.append(res.wallXphi0F * np.conj(res.wallXphi0F))
        self.wallXFourierSquare0, self.wallXFourierSquare0Err  = jackknife(resTot, nBoot)

    def corFourierPhi(self, nTherm, nBoot, decim):
        corrTot = []
        for fn in self.fnList:
            res = ConfResults(fn, nTherm,decim)
            print(fn)
            res.wallXFourierNorm()
            Nx = len(res.wallXphiNormF[0])
            corrTot.append([res.computeCttp(np.conj(res.wallXphiNormF[:,k]),res.wallXphiNormF[:,k]) for k in range(Nx)])
            print(len(corrTot[-1][0]))

        self.corFourierPhiMean, self.corFourierPhiErr  = bootstrap(corrTot, nBoot)
    def save(self, fn):
            np.savetxt(fn,[self.corFourierPhiMean, self.corFourierPhiErr])
'''
