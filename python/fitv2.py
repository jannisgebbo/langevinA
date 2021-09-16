import random
import h5py
import numpy as np
from numpy import linalg as LA
from iminuit import Minuit
from measurements import *
import matplotlib.pyplot as plt


#Functions
def axialprop(omega,parameters):
    (mp,amplitude,gammap) = parameters
    numerator= amplitude * gammap * mp**2
    denominator= np.square(-np.square(omega)+np.square(mp))+np.square(omega*gammap)
    return numerator/denominator

def phiprop(omega, parameters):
    (mp,amplitude,gammap) = parameters
    numerator= amplitude*gammap*np.square(omega) / mp**2
    denominator= np.square(-np.square(omega)+np.square(mp))+np.square(omega*gammap)
    return numerator/denominator
    
def phi0prop(omega, parameters):
    (msigma,gamma) = parameters
    numerator= 2 *gamma
    denominator= np.square(omega)+np.square(np.square(msigma)*gamma)
    return numerator/denominator
#classes

class Chi2:
    def __init__(self):
        self.z = np.empty(0)
        self.resToAdd = []
        return
    def add(self, statRes):
        self.resToAdd.append(statRes)
    
    def computeWeight(self, data, err, prediction):
        return (data - prediction) / err
    
    def __call__(self,  listFunc, realOrImag = np.real):
        z = np.zeros(len(self.resToAdd[0].mean))
        for i in range(len(listFunc)):
            z += self.computeWeight(realOrImag(self.resToAdd[i].mean), realOrImag(self.resToAdd[i].err), listFunc[i])
        return np.sum(np.square(z))    

    
class Fitter:
    def __init__(self, data):
        obs = ["OtOttpFourier"]
        
        
        self.fits = dict()
        self.ndof = dict()
        self.averagechi2 = dict()
        self.averagechi2reduce = dict()
        self.func = dict()
        self.par = dict()
        self.parErr = dict()
        self.parName = dict()
        self.parLims = dict()
        
        for o in obs:
            self.fits[o] = dict()
            self.ndof[o] = dict()
            self.averagechi2[o] = dict()
            self.averagechi2reduce[o] = dict()
            self.func[o] = dict()
            self.par[o] = dict()
            self.parErr[o] = dict()
            self.parName[o] = dict()
            self.parLims[o] = dict()

            
        self.data = data
                
        # This is a mapping between the fit key on the left and the actual quantity (ies) we are fitting. Allow to define combined fit in this way.
        self.baseKeys = dict()
        self.baseKeys["A"] = ["A"]
        self.baseKeys["phi"] = ["phi"]
        self.baseKeys["dphi"] = ["dphi"]
        self.baseKeys["phi0"] = ["phi0"]
        self.baseKeys["Aphi"] = ["A", "phi"]
        
        self.xs = dict()
        
        
        
        # Actual definition of the function to fit, per observable and per fit key. The lambda function should receive x
        # and the fit parameters. It should return a list of array, each array corresponding to the function to be fitted
        # evaluated at x, one per observable fitted (so one for a simple fit and more for combined fit, equal to the length of baseKeys).
        # Can define a fit for any observable which is also defined in the measurements; we use meas.getObs(...) to retriee the data.
        
        
        #TODO: Define the static correlators at this level. Can define
        
        self.func["OtOttpFourier"]["A"] = lambda x, par : [axialprop(x, par)]
        self.func["OtOttpFourier"]["phi"] = lambda x, par : [phiprop(x, par)]
        self.func["OtOttpFourier"]["dphi"] = lambda x, par : [phiprop(x, par)]
        self.func["OtOttpFourier"]["phi0"] = lambda x, par : [phi0prop(x, par)]
        
        self.func["OtOttpFourier"]["Aphi"] = lambda x, par : [axialprop(x, (par[0],par[1],par[3])), phiprop(x, (par[0],par[2],par[3]))]
        

        # Below we define the size, name and limits of the parameters, per observables and per fitKeys.
        
        self.par["OtOttpFourier"]["A"] = np.zeros(3)
        self.parName["OtOttpFourier"]["A"] = ["mp","amplitudeA","gammap"]
        self.parLims["OtOttpFourier"]["A"] = [(0, None), (0, None),(0, None)]
        
        self.par["OtOttpFourier"]["phi"] = np.zeros(3)
        self.parName["OtOttpFourier"]["phi"] = ["mp","amplitudephi","gammap"]
        self.parLims["OtOttpFourier"]["phi"] = [(0, None), (0, None),(0, None)]
        
        self.par["OtOttpFourier"]["dphi"] = np.zeros(3)
        self.parName["OtOttpFourier"]["dphi"] = ["mp","amplitudephi","gammap"]
        self.parLims["OtOttpFourier"]["dphi"] = [(0, None), (0, None),(0, None)]
        
        
        self.par["OtOttpFourier"]["Aphi"] = np.zeros(4)
        self.parName["OtOttpFourier"]["Aphi"] = ["mp","amplitudeA","amplitudephi","gammap"]
        self.parLims["OtOttpFourier"]["Aphi"] = [(0, None), (0, None),(0, None),(0, None)]
        

        
    def setParValues(self, obs, key, pars):
        self.par[obs][key] = pars
      
    def fit(self, obs, key):                                      
        chi2 = Chi2()
        
        for k in self.baseKeys[key]:
            (x, fx) = self.data.getObs(obs, k)
            chi2.add(fx)
            self.xs[obs] = x
                                           
        self.fits[obs][key] = Minuit(lambda x: chi2(self.func[obs][key]( self.xs[obs], x)), self.par[obs][key], name=self.parName[obs][key])
        
        
        self.fits[obs][key].errordef = Minuit.LEAST_SQUARES
        
        self.fits[obs][key].limits = self.parLims[obs][key]
                
        self.fits[obs][key].migrad()
        self.fits[obs][key].minos()
                
        self.ndof[obs][key] = len(self.baseKeys[key]) * len(self.xs[obs]) - len(self.par[obs][key])
        self.averagechi2[obs][key] = self.fits[obs][key].fval
        self.averagechi2reduce[obs][key] = self.fits[obs][key].fval / self.ndof[obs][key]
        
        c = 0
        for v in self.fits[obs][key].values:
            self.par[obs][key][c] = v
            c += 1   
            
        c = 0
        self.parErr[obs][key] = np.zeros(len(self.par[obs][key]))
        for v in self.fits[obs][key].errors:
            self.parErr[obs][key][c] = v
            c += 1   
        
        return (self.averagechi2[obs][key], self.averagechi2reduce[obs][key], self.ndof[obs][key], self.fits[obs][key].values)

    
    def plot(self, obs, key, color="b"):
        if obs in self.xs.keys():
            x = self.xs[obs]
        else:
            x = self.data.getObs(obs, key, withY = False)
        
        print(self.par[obs][key])
        prediction = self.func[obs][key](x, self.par[obs][key])
        for p in prediction:
            plt.plot(x, p, color)