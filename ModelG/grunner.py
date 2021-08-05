#!/usr/bin/env python3
import os
import subprocess
import json 
import random
import datetime

random.seed()

# set the overall tag
data = {
    # lattice dimension
    "NX" : 32,

    # Time stepping
    "finaltime" : 10,
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
    "chi" : 2.,
    "seed" : 122335456,
    "restart" : "false",

    #initial condition"
    "outputfiletag" : "grun",
    "saveFrequencyInTime" : 0.8,
}

# output is prepended with tag_...... For example if tag is set to "foo". Then
# all outputs are of the form "foo_averages.txt"
tag = "default"

# dump the data into an input file
def datatoinput():
    with open(data["outputfiletag"] + '.in','w') as fh:
        for key, value in data.items() :
            fh.write("%s = %s\n" %(key, value) )
    
# dump the data into a .json file
def datatojson():
    with open(data["outputfiletag"]+ '.json', 'w') as outfile:
        json.dump(data,outfile,indent=4)


# Canonicalize the names for a given set of parameters
def getdefault_filename() :
    name = "%s_N%03d_m%08d_h%06d_c%05d" % (tag,data["NX"],round(100000*data["mass"]),round(1000000*data["H"]),round(100*data["chi"]))
    return name

# Canonicalize the names for a given set of parameters, with a scan in m2
def getdefault_filename_m2change() :
    s = "xxxxxxxxxxxxx"
    name = "%s_N%03d_m%.8s_h%06d_c%05d" % (tag,round(data["NX"]),s,round(1000000*data["H"]),round(100*data["chi"]))
    return name

# Canonicalize the names for a given set of parameters, with scan in H
def getdefault_filename_Hchange() :
    s = "xxxxxxxxxxxxx"
    name = "%s_N%03d_m%08d_h%.6s_c%05d" % (tag,data["NX"],round(100000*data["mass"]),s,round(100*data["chi"]))
    return name

# Canonicalize the names for a given set of parameters, with scan in N
def getdefault_filename_Nchange() :
    s = "xxxxxxxxxxxxx"
    name = "%s_N%.3s_m%08d_h%06d_c%05d" % (tag,s,round(100000*data["mass"]),round(1000000*data["H"]),round(100*data["chi"]) )
    return name

# Canonicalize the names for a given set of parameters, with scan in chi
def getdefault_filename_chichange() :
    s = "xxxxxxxxxxxxx"
    name = "%s_N%03d_m%08d_h%06d_c%.5s" % (tag,data["NX"],round(100000*data["mass"]),round(1000000*data["H"]),s)
    return name


# Sets the filename to getdefaultname
def setdefault_filename() :
    data["outputfiletag"] = getdefault_filename()


# Runs on cori regular que  with time in hours. One should set dry_run=False to
# actually run the code
def corirun(time=2, debug=False, shared=False, dry_run=True, moreopts=[], seed=None) :
    filenamesh = data["outputfiletag"] + '.sh'
    with open(filenamesh,'w') as fh:
        print("#!/bin/bash",file=fh) 
        if debug :
            print("#SBATCH -q debug",file=fh) 
            print("#SBATCH -t 00:10:00",file=fh) 
            print("#SBATCH -N 1",file=fh) 
            print("#SBATCH --ntasks=32",file=fh) 
            print("#SBATCH --cpus-per-task=2",file=fh) 
        elif shared :
            print("#SBATCH -q shared",file=fh) 
            print("#SBATCH -t %s" % (str(round(time*60))),file=fh) 
            print("#SBATCH --ntasks=8",file=fh) 
            print("#SBATCH --cpus-per-task=2",file=fh) 
        else:
            print("#SBATCH -q regular",file=fh) 
            print("#SBATCH -t %s" % (str(round(time*60))),file=fh) 
            print("#SBATCH -N 1",file=fh) 
            print("#SBATCH --ntasks=32",file=fh) 
            print("#SBATCH --cpus-per-task=2",file=fh) 
        print("#SBATCH -C haswell",file=fh) 

        print("",file=fh) 
        print("module load gsl",file=fh) 
        print("module load cray-petsc",file=fh) 
        print("",file=fh) 
        print("#run the application:",file=fh) 

        print('date  "+%%x %%T" > %s_time.out' % (data["outputfiletag"]),file=fh) 
        # get the program
        path = os.path.abspath(os.path.dirname(__file__))
        prgm = path + "/SuperPions.exe"
        # set the seed and the inputfile
        if seed is None:
            data["seed"] = random.randint(1,2000000000)
        else:
            data["seed"] = seed
        # write the data to an inputfile
        datatoinput()
        # write the data to an .json
        datatojson()

        #write the command that actually runds the program
        print("srun --cpu_bind=cores %s input=%s" % (prgm,data["outputfiletag"]+'.in'), end=' ', file=fh) 
        for opt in moreopts:
            print(opt,end=' ', file=fh)
        print(file=fh)
        print('date  "+%%x %%T" >> %s_time.out' % (data["outputfiletag"]),file=fh) 

    if not dry_run:
        subprocess.run(['sbatch',filenamesh])

#runs the actual command current value of data  with mpiexec
def run(moreopts=[], dry_run=True, time=0, seed=None, ncpus="4") :
    # find the program
    path = os.path.abspath(os.path.dirname(__file__))
    prgm = path + "/SuperPions.exe"

    # set the seed and the inputfile
    if seed is None:
        data["seed"] = random.randint(1,2000000000)
    else:
        data["seed"] = seed

    datatoinput()
    datatojson()

    # Execute the program
    opts = ["mpiexec","-n", ncpus, prgm, "input="+data["outputfiletag"] + '.in'] 
    opts.extend(moreopts)
    print(opts)
    if not dry_run:
        subprocess.run(opts)

if __name__ == "__main__":
    print(getdefault_filename())
    print(getdefault_filename_Nchange())
    print(getdefault_filename_m2change())
    print(getdefault_filename_Hchange())
    print(getdefault_filename_chichange())
    setdefault_filename()
    run(dry_run=True)
    corirun(dry_run=True, time=0.25)
