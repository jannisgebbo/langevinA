import h5py
import numpy as np
import errno
import os.path


# Compute the correlator in time between arr1 and arr2.
#
# The correlation function is computed over the full range if nTMax is not
# passed as an optional argument. If nTMax is passd then the maximum range of
# the correlator is between 0...nTMax - 1.
#
# The routine will also substract the connected part by setting conn (for
# connected) to True, but this is false by default
#
# The statistical method for computing the average is passed as the third
# arguement.  The default is
#
# lambda x: (np.mean(x), 0)
#
# Examples:
#
# computeOtOtp(d, np.conj(d))
def computeOtOtp(arr1, arr2, statFunc=lambda x: (np.mean(x), 0), conn=False, nTMax=-1, decim=1):

    # Take the full possible range if nTMax is not passed
    Npoints = len(arr1)
    if nTMax == -1:  # nTMax was not passed
        nTMax = Npoints

    # Allocate space for the result
    Cttp = np.zeros(nTMax, dtype=arr1.dtype)
    CttpErr = np.zeros(nTMax, dtype=arr1.dtype)

    # Compute means if necessary for connected part
    if conn:
        av1 = np.mean(arr1)
        av2 = np.mean(arr2)

    # For each time compute the product of a1 and rolled a2
    # an average of this is the correlation ofunction
    for tt in range(nTMax):
        tmp = arr1 * np.roll(arr2, -tt)[0:len(arr1)]
        if conn:
            tmp -= av1 * av2

        Cttp[tt], CttpErr[tt] = statFunc(tmp[0:Npoints - tt:decim])

    return Cttp, CttpErr


# Compute the O(t) O(0) correlator over blocks in time.
#
# Inputs:
#
# array1 and array2, as well as the dimension of the correlation array, nTMax.
# steps is the nuber of array entries used to form a block average
#
# Outputs:
#
# A  python array,  containing the O(t) O(0) as an array, for each block.
#
# [ [data], [data], [data], ..... ]
def computeBlockedOtOtp(arr1, arr2, nTMax, steps, decim=1, conn=False):
    res = []
    sMax = len(arr1) - steps
    for s in range(0, sMax, steps):
        tmpR, tmpE = computeOtOtp(arr1[s:s + steps],
                                  arr2[s:s + steps],
                                  lambda x: (np.mean(x), 0),
                                  nTMax=nTMax,
                                  decim=decim,
                                  conn=conn)
        res.append(tmpR)

    return np.asarray(res)


def blockedOtOtp(arr1, arr2, nTMax, nblocks):
    res = []

    sMax = len(arr1) - nTMax
    steps = int(sMax/nblocks)
    sMax = steps * nblocks

    for s in range(0, sMax, steps):
        tmpR, tmpE = computeOtOtp(arr1[s:s + steps],
                                  arr2[s:s + steps + nTMax],
                                  lambda x: (np.mean(x), 0),
                                  nTMax=nTMax,
                                  decim=1,
                                  conn=False)
        res.append(tmpR)

    return np.asarray(res)


################################################################################
class Loader:

    # Initialize the loader by checking if filename exists
    def __init__(self, filename, starttime=0, removemean=False, nblocks=1):

        if not os.path.exists(filename):
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        self.filename = filename
        self.datadir = os.path.dirname(filename)
        self.starttime = starttime
        self.removemean = removemean
        self.ntime_total, self.nx = self._compute_dimensions()
        self.nblocks = nblocks

        self._base_keys = {"phi0": 0, "phi1": 1, "phi2": 2, "phi3": 3, "A1": 4,
                           "A2": 5, "A3": 6, "V1": 7, "V2": 8, "V3": 9, "phiNorm": 10}

        self.loaders = {}
        self.keygroups = {}
        self._fill_loaders()
        self._fill_keygroups()

    @property
    def nblocks(self):
        return self._nblocks

    @nblocks.setter
    def nblocks(self, value):
        if value < 1:
            raise ValueError("The number of blocks must be at least one")
        self._nblocks = value
        self._blocksize = int((self.ntime_total - self.starttime)/self.nblocks)

    @property
    def blocksize(self):
        return self._blocksize

    # Helper function defining when the block starts.
    #
    # start = blockstart(i)
    # stop = blockstart(i+1)
    #
    # If iblock is greater than nblocks-1, returns None
    def blockstart(self, iblock):
        if iblock < 0:
            return self.starttime
        elif iblock < self.nblocks:
            return self.starttime + iblock*self.blocksize
        else:
            return None

    # Load specified data into numpy array
    #
    # Examples:
    #
    # ld.load("phi0")   #uses the class value of removemean
    # ld.load("phi0", removemean=True)
    # ld.load("phi1_kx", k=2)  # load k=2 values from k
    # ld.load("phi1_x")  # loads the x walls all blocks
    # ld.load("phi1_y", block=2)  #loads the second block of y walls
    # ld.load("A3_kz", k=1) # load the A3 k=1 in z
    #
    # For each key this routine simply calls the corresponding loader
    # You can add your own loader if one doesn't fit your needs.
    #
    # ld.loader["myfield"] = myloadingfunc(self, key, **kwargs)
    #
    # The keygroups dictionary exists to loop over groups of related keys.
    #
    # Thus the key "A123" in keygroups["A123"] contains the list ["A1","A2","A3"]
    #
    # for key in ld.keygroups["A123"] :
    #     data = ld.load(key, k=2)
    #
    def load(self, key, **kwargs):
        return self.loaders[key](self, key, **kwargs)

    # Load a column of a given key for k=0:
    #
    # Possible keys are:
    #
    #  phi0, phi1, phi2, phi3, A1, A2, A3, V1, V2, V3, phiNorm
    #
    # If removmean is True then the mean is subtracted from the data column.
    # upon reading. This function is added to the loaders structure
    def read_column_k0(self, key, **kwargs):

        akeys = self._base_keys

        h5 = h5py.File(self.filename, 'r')
        data = np.asarray(h5["phi"][self.starttime:, akeys[key]])

        removemean = self.removemean
        if "removemean" in kwargs:
            removemean = kwargs["removemean"]

        # Remove the mean from the specified axis
        if self.removemean:
            mean = np.mean(data, axis=0)
            data -= mean

        return data

    # Load a column for a column of the form
    #
    #  phi0_kx or A1_kz
    #
    # Examples implemented:
    #
    # load("phi0_kx", k=2) # uses the default class value of removemean
    # load("phi0_kx")   # reads k=0
    # load("phi0_kx", k=2, removemean=True) #overrides the default remove mean by class
    def read_column_k(self, key,  **kwargs):

        filename_fourier = os.path.splitext(self.filename)[0] + "_out.h5"
        if not os.path.exists(filename_fourier):
            raise FileNotFoundError(errno.ENOENT, os.strerror(
                errno.ENOENT), filename_fourier)

        wkeys = self._base_keys

        try:
            name, tag = key.split("_")
        except:
            print("read_column_k: Unable to process key %s" % (key))
            raise KeyError

        # Get an integer for the fourier mode
        k = kwargs.get("k", 0)
        direction = {"kx": "x", "ky": "y", "kz": "z"}
        dsetname = "wall" + direction[tag] + "_k"
        h5 = h5py.File(filename_fourier, 'r')

        data = np.ravel(np.asarray(
            h5[dsetname][self.starttime:, wkeys[name], k, :]).view(dtype=np.complex128))

        removemean = self.removemean
        if "removemean" in kwargs:
            removemean = kwargs["removemean"]

        if removemean:
            mean = np.mean(data, axis=0)
            data -= mean

        return data

    # Load a wall of the form with a key of the forms, e.g.
    #
    #  phi0_x, A1_z
    #
    # Examples:
    #
    # load("phi0_x")  # load all data
    # load("phi0_x", block=2)  # load only block 2

    def read_wall(self, key,  **kwargs):
        wkeys = self._base_keys

        try:
            name, tag = key.split("_")
        except:
            print("read_wall: Unable to process key %s" % (key))
            raise KeyError

        if not tag in {"x", "y", "z"}:
            raise KeyError("read_wall: bad direction %s" % (tag))

        dsetname = "wall" + tag
        h5 = h5py.File(self.filename, 'r')

        if kwargs.get('block', None) is not None:
            start = self.blockstart(kwargs["block"])
            stop = self.blockstart(kwargs["block"]+1)
        else:
            start = self.starttime
            stop = None

        data = np.asarray(h5[dsetname][start:stop, wkeys[name], :])
        if data.nbytes > 100000000:  # 100 Meg
            size = data.nbytes/1000000
            print("Warning the loaded array is greater than 100 M. Read by blocks, e.g. load(\"phi0_x\", block=3)\n\tThe current size is %f M\n" % (size))

        return data

    # Helper function for filling up the loaders structure
    def _fill_loaders(self):
        for key in self._base_keys:
            self.loaders[key] = Loader.read_column_k0
        for key in self._base_keys:
            self.loaders[key + "_kx"] = Loader.read_column_k
        for key in self._base_keys:
            self.loaders[key + "_ky"] = Loader.read_column_k
        for key in self._base_keys:
            self.loaders[key + "_kz"] = Loader.read_column_k
        for key in self._base_keys:
            self.loaders[key + "_x"] = Loader.read_wall
        for key in self._base_keys:
            self.loaders[key + "_y"] = Loader.read_wall
        for key in self._base_keys:
            self.loaders[key + "_z"] = Loader.read_wall

    # Helper function for producing groups of keys e.g.
    #
    # make_keygroup("phi",["1","2"],["_kx","_ky","_kz"]) produces
    #
    # a list of six keys  of the form, [phi1_kx, ...., phi2_kz]
    def _make_keygroup(base, x123, kxyz=None):
        keys = []
        for nmb in x123:
            base_with_number = base + nmb
            if kxyz:
                for kx in kxyz:
                    keys.append(base_with_number + kx)
            else:
                keys.append(base_with_number)
        return keys

    # Helper function for producing groups of keys
    def _fill_keygroups(self):

        # Add the real groups of keys
        x123 = ["1", "2", "3"]
        kxyz = ["_kx", "_ky", "_kz"]
        xyz = ["_x", "_y", "_z"]
        self.keygroups = {
            "phi0123": Loader._make_keygroup("phi", ["0"]+x123),
            "phi123": Loader._make_keygroup("phi", x123),
            "A123": Loader._make_keygroup("A", x123),
            "V123": Loader._make_keygroup("V", x123),
            # momentum space variety
            "phi0123_kxyz": Loader._make_keygroup("phi", ["0"]+x123, kxyz),
            "phi123_kxyz": Loader._make_keygroup("phi", x123, kxyz),
            "A123_kxyz": Loader._make_keygroup("A", x123, kxyz),
            "V123_kxyz": Loader._make_keygroup("V", x123, kxyz),
            "AV123_kxyz": Loader._make_keygroup("A", x123, kxyz)
            + Loader._make_keygroup("V", x123, kxyz),
            # xyz variety
            "phi0123_xyz ": Loader._make_keygroup("phi", ["0"]+x123, xyz),
            "phi123_xyz": Loader._make_keygroup("phi", x123, xyz),
            "A123_xyz": Loader._make_keygroup("A", x123, xyz),
            "V123_xyz": Loader._make_keygroup("V", x123, xyz),
        }

        # Every key in loaders forms its own keygroup
        for key in self.loaders:
            self.keygroups[key] = [key]

    # opens the h5 and determine the dimensions
    def _compute_dimensions(self):
        h5 = h5py.File(self.filename, 'r')
        ntime_total, nfields, nx = h5["wallx"].shape
        return ntime_total, nx


################################################################################
class TimeCorrelator:

    def __init__(self, loader, nblocks=10, nTMax=2000):
        self.loader = loader
        self.nTMax = nTMax
        self.nblocks = nblocks

    def correlate_key(self, key, **kwargs):
        if key in self.loader.keygroups:
            return self.correlate_keys(*self.loader.keygroups[key], **kwargs)
        else:
            raise KeyError(key)

    def correlate_keys(self, *keys, **kwargs):
        nblocks = self.nblocks
        first = True
        for key in keys:
            d1 = self.loader.load(key, **kwargs)
            d2 = np.conj(d1)

            if first:
                # OtOtp = computeBlockedOtOtp(d1, d2, self.nTMax, self.loader.blocksize)
                OtOtp = blockedOtOtp(d1, d2, self.nTMax, nblocks)
                first = False
            else:
                # OtOtp = np.vstack((OtOtp,
                #     computeBlockedOtOtp(d1, d2, self.nTMax, self.loader.blocksize)))
                OtOtp = np.vstack(
                    (OtOtp, blockedOtOtp(d1, d2, self.nTMax, nblocks)))

        return (np.mean(OtOtp.real, axis=0), np.std(OtOtp.real, axis=0, ddof=1),
                OtOtp.real)

################################################################################


class StaticCorrelator:

    def __init__(self, loader):
        self.loader = loader
        return

    def correlate_key(self, key, **kwargs):
        if key in self.loader.keygroups:
            return self.correlate_keys(*self.loader.keygroups[key], **kwargs)
        else:
            raise KeyError(key)

    def correlate_keys(self, *keys, **kwargs):
        nblocks = self.loader.nblocks
        first = True
        for key in keys:
            for ib in range(0, nblocks):

                data = self.loader.load(key, block=ib, **kwargs)

                if first:
                    OxOy = StaticCorrelator.correlate_data(data)
                    first = False
                else:
                    OxOy = np.vstack(
                        (OxOy, StaticCorrelator.correlate_data(data)))

        return np.mean(OxOy, axis=0), np.std(OxOy, axis=0, ddof=1), OxOy

    @staticmethod
    def correlate_data(d):
        row = np.zeros(d.shape[1], dtype=d.dtype)
        d1 = d
        for i in range(0, d.shape[1]):
            row[i] = np.sum(d*d1)/d.size
            d1 = np.roll(d1, 1, axis=1)
        return row
