#!/usr/bin/env python3
import os
import subprocess
import json
import random
import dstack
import datetime
import math


GLOBAL_PETSCPKG_PATH_SEAWULF="${PKG_CONFIG_PATH}:/gpfs/home/adrflorio/petsc/arch-linux2-c-debug/lib/pkgconfig/"

random.seed()

# set the overall tag
data = {
    # lattice dimension
    "NX": 32,

    # Time stepping
    "finaltime" : 10,
    "initialtime" : 0,
    "deltat" : 0.24,

    #Action
    "mass0" : -4.70052,
    "dmassdt" : 0 ,

    "lambda" : 4.,
    "H" :0.003,
    "chi" : 5.,
    "gamma" : 1.,
    "diffusion" : 0.3333333,

    #initial condition"
    "evolverType" : "PV2HBSplit23",
    "seed" : 122335456,
    "restart" : False,
    "outputfiletag" : "grun",
    "saveFrequency" : 3,
    "thermalization_time" : 0.0 ,

    # For running multi-events 
    "eventmode": False,
    "nevents" : 1,
    "last_stored_event": -1,
    "diffusiononly": False 
}


# output is prepended with tag_...... For example if tag is set to "foo". Then
# all outputs are of the form "foo_averages.txt"
tag = "default"


# dump the data into a .json file
def datatojson():
    with open(data["outputfiletag"] + '.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)


# Canonicalize the names for a given set of parameters
def getdefault_filename():
    name = "%s_N%03d_m%08d_h%06d_c%05d" % (tag, data["NX"], round(
        100000*data["mass0"]), round(1000000*data["H"]), round(100*data["chi"]))
    return name


# Canonicalize the names for a given set of parameters, with a scan in m2
def getdefault_filename_m2change():
    s = "xxxxxxxxxxxxx"
    name = "%s_N%03d_m%.8s_h%06d_c%05d" % (tag, round(
        data["NX"]), s, round(1000000*data["H"]), round(100*data["chi"]))
    return name


# Canonicalize the names for a given set of parameters, with scan in H
def getdefault_filename_Hchange():
    s = "xxxxxxxxxxxxx"
    name = "%s_N%03d_m%08d_h%.6s_c%05d" % (tag, data["NX"], round(
        100000*data["mass0"]), s, round(100*data["chi"]))
    return name


# Canonicalize the names for a given set of parameters, with scan in N
def getdefault_filename_Nchange():
    s = "xxxxxxxxxxxxx"
    name = "%s_N%.3s_m%08d_h%06d_c%05d" % (tag, s, round(
        100000*data["mass0"]), round(1000000*data["H"]), round(100*data["chi"]))
    return name


# Canonicalize the names for a given set of parameters, with scan in chi
def getdefault_filename_chichange():
    s = "xxxxxxxxxxxxx"
    name = "%s_N%03d_m%08d_h%06d_c%.5s" % (tag, data["NX"], round(
        100000*data["mass0"]), round(1000000*data["H"]), s)
    return name


# Sets the filename to getdefaultname
def setdefault_filename():
    data["outputfiletag"] = getdefault_filename()

########################################################################
def find_program(program_name="SuperPions.exe"):
    # find the program
    path = os.path.abspath(os.path.dirname(__file__))
    return path + "/" + program_name


#########################################################################
# Runs on cori
#########################################################################
def corirun(time=2, debug=False, shared=False, dry_run=True, moreopts=["-log_view"], seed=None, nnodes=1, parallel=False, environment=[],shellname=None):

    prgm = find_program()

    tag = data["outputfiletag"]

    # Create a run directory if does not exist, and cd to it
    dstack.pushd(tag,mkdir=True)

    if shellname:
        filenamesh = shellname + '.sh'
    else:
        filenamesh = tag + '.sh'

    fh = open(filenamesh, 'w')

    print("#!/bin/bash", file=fh)
    if debug:
        print("#SBATCH -q debug", file=fh)
        print("#SBATCH -t 00:10:00", file=fh)
        print("#SBATCH -N {}".format(nnodes), file=fh)
        print("#SBATCH --ntasks={}".format(nnodes*32), file=fh)
        print("#SBATCH --cpus-per-task=2", file=fh)
    elif shared:
        print("#SBATCH -q shared", file=fh)
        print("#SBATCH -t {}".format(int(math.ceil(time*60.))), file=fh)
        print("#SBATCH --ntasks=8", file=fh)
        print("#SBATCH --cpus-per-task=2", file=fh)
    else:
        print("#SBATCH -q regular", file=fh)
        print("#SBATCH -t {}".format(int(math.ceil(time*60.))), file=fh)
        print("#SBATCH -N %d" % (nnodes), file=fh)
        print("#SBATCH --ntasks={}".format(nnodes*32), file=fh)
        print("#SBATCH --cpus-per-task=2", file=fh)
    print("#SBATCH -C haswell", file=fh)

    #
    # Write header portion of the batch file
    #
    print("", file=fh)
    print("module load gsl", file=fh)
    print("module load cray-petsc", file=fh)
    print("module load cray-hdf5-parallel", file=fh)
    print("export HDF5_DISABLE_VERSION_CHECK=2", file=fh)

    print("", file=fh)

    #
    # run the program
    #
    print("#run the application:", file=fh)

    print('date  "+%%x %%T" > %s_time.out' %
          (data["outputfiletag"]), file=fh)
    # set the seed and the inputfile
    if seed is None:
        data["seed"] = random.randint(1, 2000000000)
    else:
        data["seed"] = seed
    # write the data to an .json
    datatojson()

    # write the command that actually runds the program
    print("srun --cpu_bind=cores %s -input %s " %
          (prgm, data["outputfiletag"]+'.json'), end=' ', file=fh)
    for opt in moreopts:
        print(opt, end=' ', file=fh)
    print(file=fh)

    print('date  "+%%x %%T" >> %s_time.out' %
          (data["outputfiletag"]), file=fh)

    fh.close()

    if not dry_run:
        subprocess.run(['sbatch', filenamesh])

    # return to the root directory
    dstack.popd()

#########################################################################
# Runs on seawulf  with time in batch time. One should set dry_run=False to
# actually run the code
#########################################################################
def seawulfrun(time="00:02:00", debug=False, shared=False, dry_run=True, moreopts=[]) :
    nprocesses=24
    filenamesh = data["outputfiletag"] + '.sh'
    with open(filenamesh,'w') as fh:
        print("#!/bin/bash",file=fh) 
        if debug :
            print("#SBATCH -p debug-{}core".format(nprocesses),file=fh) 
            print("#SBATCH --time=00:10:00",file=fh) 
            print("#SBATCH --nodes=1",file=fh) 
            print("#SBATCH --ntasks-per-node={}".format(nprocesses),file=fh) 
        else:
            print("#SBATCH -p long-{}core".format(nprocesses),file=fh) 
            print("#SBATCH --time={}".format(time),file=fh) 
            print("#SBATCH --nodes=1",file=fh) 
            print("#SBATCH --ntasks-per-node={}".format(nprocesses),file=fh) 

        print("",file=fh) 
        print("module load shared",file=fh) 
        print("module load gcc-stack",file=fh) 
        print("module load hdf5/1.10.5-parallel",file=fh) 
        print("module load fftw3",file=fh) 
        print("module load cmake",file=fh) 
        print("module load gsl",file=fh)
        print("export PKG_CONFIG_PATH={}".format(GLOBAL_PETSCPKG_PATH_SEAWULF), file=fh) 
        print("export MV2_ENABLE_AFFINITY=0",file=fh)
        print("",file=fh) 
        print("#run the application:",file=fh) 

        print('date  "+%%x %%T" > %s_time.out' % (data["outputfiletag"]),file=fh) 
        # get the program
        path = os.path.abspath(os.path.dirname(__file__))
        prgm = path + "/SuperPions.exe"
        # set the seed and the inputfile
        data["seed"] = random.randint(1,2000000000)

        # Write the data to an .json
        datatojson()

        #write the command that actually runds the program
        basename = "./" + os.path.basename(data["outputfiletag"])
        print("mpirun -n {} {} -input {} ".format(nprocesses,prgm, basename +'.json'), end=' ', file=fh)
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

def pmakefiles(ncpus, seed=None):
    """Create a set of inputfiles, tag_0000, tag_0001, ...., with separate
    seeds"""
    tag = data["outputfiletag"]

    listname = tag + "_list.txt"
    fh = open(listname, "w")
    seedlist = []
    for i in range(0, int(ncpus)):
        while True:
            iseed = random.randint(1, 2000000000)
            if iseed not in seedlist:
                seedlist.append(iseed)
                break

    for i in range(0, int(ncpus)):
        if seed is None:
            data["seed"] = seedlist[i]
        else:
            data["seed"] = seed + i
        data["outputfiletag"] = tag + "_%04d" % (i)
        datatojson()
        fh.write("%s.json\n" % data["outputfiletag"])
    fh.close()

    # restore the output
    data["outputfiletag"] = tag


# Run the code in embarassingly parallel mode with gnu-parallel
def prun(moreopts=[], dry_run=True, debug=True, time=0, seed=None, ncpus="4"):
    prgm = find_program()

    tag = data["outputfiletag"]
    dstack.pushd(tag,mkdir=True)

    listname = tag + "_list.txt"
    pmakefiles(ncpus, seed)

    cmd = "cat %s | parallel %s -input {} -log_view > %s.log" % (
        listname, prgm, tag)
    print(cmd)
    if not dry_run:
        os.system(cmd)
    dstack.popd()


########################################################################
# runs the program with current value of data  and mpiexec on local
# mac.
########################################################################
def run(program_name = "SuperPions.exe", moreopts=[], dry_run=True, time=0, seed=None, ncpus="2",log_view=True):

    prgm = find_program(program_name)
    tag = data["outputfiletag"]

    # Go to the directory 
    dstack.pushd(tag,mkdir=True)

    # set the seed and the inputfile
    if seed is None:
        data["seed"] = random.randint(1, 2000000000)
    else:
        data["seed"] = seed

    datatojson()

    # Execute the program
    opts = ["mpiexec", "-n", ncpus, prgm,
            "-input",  tag + '.json']
    if log_view:
        opts.append('-log_view')
    opts.extend(moreopts)
    print(opts)
    if not dry_run:
        subprocess.run(opts)

    # Go back to the working directory
    dstack.popd()


if __name__ == "__main__":
    print(getdefault_filename())
    print(getdefault_filename_Nchange())
    print(getdefault_filename_m2change())
    print(getdefault_filename_Hchange())
    print(getdefault_filename_chichange())
    setdefault_filename()
    #corirun(dry_run=True, time=0.25)
    #corirun(dry_run=True, parallel=True, time=0.25)
    # run(dry_run=True)
    # prun(dry_run=True)
