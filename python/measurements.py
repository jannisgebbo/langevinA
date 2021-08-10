import random
import h5py
import numpy as np
from numpy import linalg as LA


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
    
    # Points over which we do the time average

    for tt in range(nTMax): #tt is the time difference
        counter = 0
        nStarts = list(range(0, Npoints - tt, decim))
        tmpArr = np.zeros(len(nStarts), dtype=type(arr1[0]))
        for t0 in nStarts: # t0 is the origin
            tmpArr[counter] = arr1[t0 + tt] * arr2[t0]
            if conn:
                tmpArr[counter] -= arr1[t0]*arr2[t0]
            counter += 1
        Cttp[tt], CttpErr[tt] = statFunc(tmpArr)
    
    return Cttp, CttpErr



# Compute the O(t) O(0) correlator over blocks in time.
def computeBlockedOtOtp(arr, nTMax, steps, pDecim=1, conn=False):
    res = []
    sMax = len(arr) - steps
    for s in range(0,sMax,steps):
        tmpR, tmpE = computeOtOtp(arr[s:s+steps],arr[s:s+steps],lambda x : (np.mean(x),0) ,nTMax=nTMax,decim=pDecim,conn=conn)
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
# TODO: Compute only half of that an symmetrize
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
    
    
    return (symmetrizeArray(oms,-1), symmetrizeArray(f),symmetrizeArray(ferr))




# For testing purpose only
def manualFourier(arr):
    Nx = len(arr)
    ks = [2 * np.pi * i /float(Nx) for i in range(Nx)]
    arrF = np.zeros(Nx, dtype=complex)
    for k in range(Nx):
        for x in range(Nx):
            arrF[k] += arr[x] * np.exp(-1j * x * ks[k] )

    return arrF



# static correlator
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
    def __init__(self,fn, thTime , dt, data_format ="new", decim =1):
        self.fn = fn
        self.thTime = thTime
        self.decim = decim
        self.dt = dt
        
        self.data_format = data_format
        
        self.wkeys = dict()
        
        if self.data_format == "old":
            self.wkeys["phi0"] = "wallX_phi_0"
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
        
        self.wallX = dict()
        self.wallXF = dict()
        self.wallXCorr = dict()
        
        self.OtOttp_blocks = dict()
        
        
        self.OtOttp_mean = dict()
        self.OtOttp_err = dict()
        self.OtOttp_time = dict()
        
        #self.OtOttpSpecFunc_blocks = dict()
        self.OtOttpSpecFunc_mean = dict()
        self.OtOttpSpecFunc_err = dict()
        self.OtOttpSpecFunc_oms = dict()
        
    def loadWallX(self,key):
        r = h5py.File(self.fn,'r')
        if self.data_format == "old":
            self.wallX[key] = np.asarray(r[self.wkeys[key]])[self.thTime:] 
        else:
            self.wallX[key] = np.asarray(r["corrx"])[self.thTime:,self.wkeys[key],:] 
            

    #TODO: Change the way we handle the average.
    def readAv(self):
        r = h5py.File(self.fn,'r')
        self.phi = np.asarray(r["phi"])[self.thTime:]
        self.phiNorm = self.phi[:,4]

    def computeMag(self, direc = 0):
        self.readAv()
        self.mag, self.magErr = bootstrap(self.phi[:,direc],100)


    #Compute the fourier transform of a given wall, specified by the appropriate key.
    def computeWallXFourier(self, key):
        self.loadWallX(key)
        Nx = len(self.wallX[key][0])
        self.wallXF[key] = np.zeros(np.shape(self.wallX[key]), dtype=complex)
        for t in range(len(self.wallX[key])):
            self.wallXF[key][t] = np.fft.fft(self.wallX[key][t]) / Nx
    #Computes the time correlator of a given key.
    def computeOtOtpBlocked(self, key, tMax, blockSizeT, errFunc, momNum = 0, conn = False):
        if not key in self.OtOttp_blocks.keys():
            if key != "phi0":
                keys = [key + str(i) for i in [1,2,3]]
                for k in keys:
                    self.computeWallXFourier(k)
                av = np.zeros(np.shape(self.wallXF[keys[0]][:,momNum]), dtype = self.wallXF[keys[0]][:,momNum].dtype)
                for k in keys:
                    av += self.wallXF[k][:,momNum]
                av /= 3.0
            else:
                self.computeWallXFourier(key)
                av = self.wallXF[key][:,momNum]


            nTMax = int(np.floor(tMax / self.dt))
            blockSize = int(np.floor(blockSizeT / self.dt))

            self.OtOttp_blocks[key] = computeBlockedOtOtp(av,nTMax, blockSize, self.decim, conn = conn)
       
        self.OtOttp_mean[key], self.OtOttp_err[key] = errFunc(self.OtOttp_blocks[key])
        
        self.OtOttp_time[key] = np.arange(0,tMax, self.dt)

    #Compute the statistical correlator of a given key.
    def computeStatCor(self, key, omMax, errFunc, filterFunc = lambda x : 1):
        try :
            if not key in self.OtOttp_blocks.keys():
                raise ;
            else:
                self.OtOttpSpecFunc_oms[key],self.OtOttpSpecFunc_mean[key],self.OtOttpSpecFunc_err[key] = toFourierBlocked(self.OtOttp_blocks[key], self.dt, omMax, errFunc, filterFunc)
        except:
            print("the two point function in time of {} was not computed. Need to call computeOtOtpBlocked first.".format(key))
            
    #TODO: improve/ check: static correlator.  
    def correlator0(self, key):
        r = h5py.File(self.fn,'r')
        self.wallX[key] = np.asarray(r[self.wkeys[key]])[self.thTime:]
        self.wallXCorr[key] = []
        for t in range(len(self.wallX[key])):
            self.wallXCorr[key].append(twoPtAv(self.wallX[key][t]))



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