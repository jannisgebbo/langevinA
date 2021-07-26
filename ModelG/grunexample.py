import grunner as grun

# set the overall tag
grun.tag="gset"
grun.data = {
    # lattice dimension
    "NX" : 32,

    # Time stepping
    "finaltime" : 2,
    "initialtime" : 0,
    "deltat" : 0.02,
    "deltatHB" : 0.02,
    "evolverType" : 7,

    #Action
    "mass" : -4.813,
    "lambda" : 4.,
    "gamma" : 1.,
    "H" :0.004,
    "sigma" : 0.666666666666,
    "seed" : 122335456,
    "chi" : 2.,

    #initial condition"
    "zeroStart" : "true",
    "outputfiletag" : "grun",
    "saveFrequencyInTime" : 0.4,
}

Ns = [24,48]
for N in Ns:
    grun.data["NX"] = N 
    grun.data["outputfiletag"]=grun.getdefault_filename()
    grun.corirun(dry_run=True, time=(N/16.)**3*0.05)
