import grunner as grun

# set the overall tag
grun.tag="gset"
grun.data = {
    # lattice dimension
    "NX" : 32,

    # Time stepping
    "finaltime" : 400,
    "initialtime" : 0,
    "deltat" : 0.04,
    "deltatHB" : 0.04,
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
    "saveFrequencyInTime" : 0.8,
}

Ns = [32]
for N in Ns:
    grun.data["NX"] = N 
    grun.data["outputfiletag"]=grun.getdefault_filename()
    grun.run(dry_run=True, time=(N/16.)**3*0.05)
