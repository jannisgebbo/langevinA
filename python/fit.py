import random
import h5py
import numpy as np
from numpy import linalg as LA
from iminuit import Minuit
from measurements import *
import matplotlib.pyplot as plt


#Functions
def axialprop(omega,mp,amplitude,gammap):
    numerator= amplitude
    denominator= np.square(-np.square(omega)+np.square(mp))+np.square(omega*gammap)
    return numerator/denominator

def phiprop(omega,mp,amplitude,gammap):
    numerator= amplitude*np.square(omega)
    denominator= np.square(-np.square(omega)+np.square(mp))+np.square(omega*gammap)
    return numerator/denominator
    
def phi0prop(omega,msigma,gamma):
    numerator= 2 *gamma
    denominator= np.square(omega)+np.square(np.square(msigma)*gamma)
    return numerator/denominator
#classes


#Work in proigress, store the results and call the corersponding measurements in a user friendly way.
class FitResult:
    def __init__(self,data):
        self.data = data
        
        self.par = dict()
        
        self.par["mp"] = 0
        self.par["ampCharge"]=0
        self.par["ampPhi"]=0
        self.par["gammap"]=0
        
        
        
    def chisquareaxial(self,parameter):
        (mp,amplitudecharge,gammap)=parameter
        expdata=np.real(self.data.OtOttpSpecFunc_mean["A"])
        prediction=axialprop(self.data.OtOttpSpecFunc_oms["A"],mp,amplitudecharge,gammap)
        error= np.real(self.data.OtOttpSpecFunc_err["A"])
        z = (expdata - prediction) / error
        return np.sum(np.square(z))
    
    def chisquarphi(self,parameter):
        (mp,amplitudephi,gammap)=parameter
        expdata=np.real(self.data.OtOttpSpecFunc_mean["phi"])
        prediction=phiprop(self.data.OtOttpSpecFunc_oms["phi"],mp,amplitudephi,gammap)
        error= np.real(self.data.OtOttpSpecFunc_err["phi"])
        z = (expdata - prediction) / error
        return np.sum(np.square(z))
    
    def modelaxial(self):
        return axialprop(self.data.OtOttpSpecFunc_oms["A"],self.par["mp"],self.par["ampCharge"],self.par["gammap"])
        
    def modelphi(self):
        return phiprop(self.data.OtOttpSpecFunc_oms["phi"],self.par["mp"],self.par["ampPhi"],self.par["gammap"])
    
    def fitAA(self):
        self.AAfit=Minuit(lambda x: self.chisquareaxial(x), (self.par["mp"],self.par["ampCharge"],self.par["gammap"]),name=("mp","amplitudecharge","gammap"))
        self.AAfit.errordef = Minuit.LEAST_SQUARES
        self.AAfit.limits=[(0, None), (0, None),(0, None)]
        self.AAfit.migrad()
        self.AAfit.minos()
        self.par["mp"]=self.AAfit.values[0]
        self.par["ampCharge"]=self.AAfit.values[1]
        self.par["gammap"]=self.AAfit.values[2]
        
    
    def fitPP(self):
        self.PPfit=Minuit(lambda x: self.chisquarphi(x), (self.par["mp"],self.par["ampPhi"],self.par["gammap"]),name=("mp","amplitudephi","gammap"))
        self.PPfit.errordef = Minuit.LEAST_SQUARES
        self.PPfit.limits=[(0, None), (0, None),(0, None)]
        self.PPfit.migrad()
        self.PPfit.minos()
        self.par["mp"]=self.PPfit.values[0]
        self.par["ampPhi"]=self.PPfit.values[1]
        self.par["gammap"]=self.PPfit.values[2]
        
    def AApplot(self,mp=None,amplitudecharge=None,gammap=None):
        if mp is not None:
            self.par["mp"]=mp
        if amplitudecharge is not None:
            self.par["ampCharge"]=amplitudecharge
        if gammap is not None:
            self.par["gammap"]=  gammap
        plt.errorbar(self.data.OtOttpSpecFunc_oms["A"], self.data.OtOttpSpecFunc_mean["A"], self.data.OtOttpSpecFunc_err["A"])
        plt.plot(self.data.OtOttpSpecFunc_oms["A"],self.modelaxial())
        
        
    def PPpplot(self,mp=None,amplitudephi=None,gammap=None):
        if mp is not None:
            self.par["mp"]=mp
        if amplitudephi is not None:
            self.par["ampPhi"]=amplitudephi
        if gammap is not None:
            self.par["gammap"]=gammap
        plt.errorbar(self.data.OtOttpSpecFunc_oms["phi"], self.data.OtOttpSpecFunc_mean["phi"], self.data.OtOttpSpecFunc_err["phi"])
        plt.plot(self.data.OtOttpSpecFunc_oms["phi"],self.modelphi())
            

    

    
