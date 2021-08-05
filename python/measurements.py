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
    

    
    return (np.mean(bootstrapArr,axis = 0), (nSamples -1.0)/nSamples * np.sqrt(sigma2))
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


def computeCttp(arr1, arr2, statFunc, conn=False, tMax=-1, decim=1):
    Npoints = len(arr1)
    if tMax==-1:
        tMax=Npoints
        
    
    Cttp = np.zeros(tMax, dtype=type(arr1[0]))
    CttpErr = np.zeros(tMax, dtype=type(arr1[0]))
    
    # Points over which we do the time average

    for tt in range(tMax): #tt is the time difference
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

def twoPtAv(arr):
    N=len(arr)
    res = np.zeros(N)
    for i in range(N):
        res[i]=0
        for j in range(N):
            res[i]+=arr[j]*arr[(i+j)%N]

    return np.asarray(res)


# For testing purpose only
def manualFourier(arr):
    Nx = len(arr)
    ks = [2 * np.pi * i /float(Nx) for i in range(Nx)]
    arrF = np.zeros(Nx, dtype=complex)
    for k in range(Nx):
        for x in range(Nx):
            arrF[k] += arr[x] * np.exp(-1j * x * ks[k] )

    return arrF




#classes

class ConfResults:
    def __init__(self,fn, thTime , decim):
        self.fn = fn
        self.thTime = thTime
        self.decim = decim
        
        self.wkeys = dict()
        
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
        
        
        self.wallX = dict()
        self.wallXF = dict()
        self.wallXCorr = dict()

    def readAv(self):
        r = h5py.File(self.fn,'r')
        self.phi = np.asarray(r["phi"])[self.thTime:]
        self.phiNorm = self.phi[:,4]

    def computeMag(self, direc = 0):
        self.readAv()
        self.mag, self.magErr = bootstrap(self.phi[:,direc],100)


    def wallXFourierNorm(self):
        r = h5py.File(self.fn,'r')
        #self.wallXphi = np.asarray([np.asarray(r["wallX_phi_0"])[thTime:],np.asarray(r["wallX_phi_1"])[thTime:],np.asarray(r["wallX_phi_2"])[thTime:],np.asarray(r["wallX_phi_3"])[thTime:]])
        self.wallXphiNorm = np.asarray(r["wallX_phi_4"])[self.thTime:]
        Nx = len(self.wallXphiNorm[0])
        ks = [2 * np.pi * i /float(Nx) for i in range(Nx)]
        self.wallXphiNormF = np.zeros(np.shape(self.wallXphiNorm), dtype=complex)
        for t in range(len(self.wallXphiNorm)):
            self.wallXphiNormF[t] = np.fft.fft(self.wallXphiNorm[t]) / Nx
            #self.wallXphiNormF[t] = self.manualFourier(self.wallXphiNorm[t])

    def computeWallXFourier(self, key):
        r = h5py.File(self.fn,'r')
        #self.wallXphi = np.asarray([np.asarray(r["wallX_phi_0"])[thTime:],np.asarray(r["wallX_phi_1"])[thTime:],np.asarray(r["wallX_phi_2"])[thTime:],np.asarray(r["wallX_phi_3"])[thTime:]])
        self.wallX[key] = np.asarray(r[self.wkeys[key]])[self.thTime:]
        Nx = len(self.wallX[key][0])
        self.wallXF[key] = np.zeros(np.shape(self.wallX[key]), dtype=complex)
        for t in range(len(self.wallX[key])):
            self.wallXF[key][t] = np.fft.fft(self.wallX[key][t]) / Nx
            #self.wallXphiNormF[t] = self.manualFourier(self.wallXphiNorm[t])


    def correlator0(self, key):
        r = h5py.File(self.fn,'r')
        self.wallX[key] = np.asarray(r[self.wkeys[key]])[self.thTime:]
        self.wallXCorr[key] = []
        for t in range(len(self.wallX[key])):
            self.wallXCorr[key].append(twoPtAv(self.wallX[key][t]))




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
