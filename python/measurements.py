import random
import h5py
import numpy as np
from numpy import linalg as LA

class ConfResults:
    def __init__(self,fn, thTime , decim):
        self.fn = fn
        self.thTime = thTime
        self.decim = decim

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
    
    def computeWallXFourier0(self):
        r = h5py.File(self.fn,'r')
        #self.wallXphi = np.asarray([np.asarray(r["wallX_phi_0"])[thTime:],np.asarray(r["wallX_phi_1"])[thTime:],np.asarray(r["wallX_phi_2"])[thTime:],np.asarray(r["wallX_phi_3"])[thTime:]])
        self.wallXphi0 = np.asarray(r["wallX_phi_0"])[self.thTime:]
        Nx = len(self.wallXphi0[0])
        self.wallXphi0F = np.zeros(np.shape(self.wallXphi0), dtype=complex)
        for t in range(len(self.wallXphi0)):
            self.wallXphi0F[t] = np.fft.fft(self.wallXphi0[t]) / Nx
            #self.wallXphiNormF[t] = self.manualFourier(self.wallXphiNorm[t])
    
    def computeWallXFourierJA1(self):
        r = h5py.File(self.fn,'r')
        self.wallXJA1 = np.asarray(r["wallX_phi_5"])[self.thTime:]
        Nx = len(self.wallXJA1[0])
        self.wallXJA1F = np.zeros(np.shape(self.wallXJA1), dtype=complex)
        for t in range(len(self.wallXJA1)):
            self.wallXJA1F[t] = np.fft.fft(self.wallXJA1[t]) / Nx
            #self.wallXphiNormF[t] = self.manualFourier(self.wallXphiNorm[t])
    def computeWallXFourierJV1(self):
        r = h5py.File(self.fn,'r')
        self.wallXJV1 = np.asarray(r["wallX_phi_8"])[self.thTime:]
        Nx = len(self.wallXJV1[0])
        self.wallXJV1F = np.zeros(np.shape(self.wallXJV1), dtype=complex)
        for t in range(len(self.wallXJV1)):
            self.wallXJV1F[t] = np.fft.fft(self.wallXJV1[t]) / Nx
            #self.wallXphiNormF[t] = self.manualFourier(self.wallXphiNorm[t])
            
    def correlator0(self):
        r = h5py.File(self.fn,'r')
        self.wallXphi0 = np.asarray(r["wallX_phi_0"])[self.thTime:]
        self.wallXphi0Corr = []
        for t in range(len(self.wallXphi0)):
            self.wallXphi0Corr.append(self.twoPtAv(self.wallXphi0[t]))
        
                
    def twoPtAv(self,arr):
        N=len(arr)
        res = np.zeros(N)
        for i in range(N):
            res[i]=0
            for j in range(N):
                res[i]+=arr[j]*arr[(i+j)%N]
                
        return np.asarray(res)
        
    def computeCttp(self, arr1, arr2, conn=False, tMax=-1):
        Npoints = len(arr1)
        if tMax==-1:
            tMax=Npoints
        Cttp = np.zeros(tMax, dtype=type(arr1[0]))

        for tt in range(tMax): #tt is the time difference
            for t0 in range(0, Npoints - tt, self.decim): # t0 is the origin
                Cttp[tt] += arr1[t0 + tt] * arr2[t0] 
                if conn:
                  Cttp[tt] -= arr1[t0]*arr2[t0]
            Cttp[tt]/=float(Npoints - tt)
        return Cttp


    def computeCttpPhi0(self,conn, tMax):
        self.CttpPhi0 = self.computeCttp(self.phi[:,0], self.phi[:,0],conn,tMax)
    def computeCttpPhiNorm(self):
        self.CttpPhiNorm = self.computeCttp(self.phiNorm, self.phiNorm)
    def computeCttpPhi0(self,conn, tMax):
        self.CttpPhi0 = self.computeCttp(self.phi[:,0], self.phi[:,0],conn,tMax)
    def computeCttpPhiNorm(self):
        self.CttpPhiNorm = self.computeCttp(self.phiNorm, self.phiNorm)

    # For testing purposes
    def manualFourier(self, arr):
        Nx = len(arr)
        ks = [2 * np.pi * i /float(Nx) for i in range(Nx)]
        arrF = np.zeros(Nx, dtype=complex)
        for k in range(Nx):
            for x in range(Nx):
                arrF[k] += arr[x] * np.exp(-1j * x * ks[k] )
    
        return arrF
    
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
    sigma2 = np.zeros(np.shape(arr[0,:]), dtype=arr.dtype)
    for n in range(nSamples):
        newArr = np.concatenate((arr[:count], arr[count+nBlock:]))
        bootstrapArr.append(np.mean(newArr,axis = 0))
        count+=nBlock
        sigma2 += (bootstrapArr[-1] - thetaBar)**2
    
    bootstrapArr = np.asarray(bootstrapArr)
    
    return (np.mean(bootstrapArr,axis = 0), (nSamples -1.0)/nSamples * np.sqrt(sigma2))
    #return (np.mean(arr,axis = 0), np.std(arr,axis = 0))
    

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
