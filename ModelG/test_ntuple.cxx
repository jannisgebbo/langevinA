#include "hdf5.h"
#include "ntuple.h"
#include <array>

int main(int argc, char **argv) {

  hid_t file_id;
  file_id = H5Fcreate("work.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  static const hsize_t N = 8;
  std::array<double, N> datain = {1, 2, 3, 4, 5, 6, 7, 8};

  std::array<size_t, 1> NN1 = {8} ;
  ntuple<1> nt1(NN1, "phi1", file_id);

  std::array<size_t, 2> NN2 = {2,4} ;
  ntuple<2> nt2(NN2, "phi2", file_id);

  std::array<size_t, 3> NN3 = {2,2,2 } ;
  ntuple<3> nt3(NN3, "phi3", file_id);

  for (size_t j = 0; j < 3; j++) {

    for (int i0=0; i0 < NN1[0] ; i0++) {
      // Seems a bit excessive in 1d
      //size_t k = nt.at({i0, i1}) ;
      nt1.row[i0] = datain[i0] ;
    }

    for (int i0=0; i0 < NN2[0] ; i0++) {
    for (int i1=0; i1 < NN2[1] ; i1++) {
      size_t k = nt2.at({i0, i1}) ;
      nt2.row[k] = datain[k] ;
    }
    }

    for (int i0=0; i0 < NN3[0] ; i0++) {
    for (int i1=0; i1 < NN3[1] ; i1++) {
    for (int i2=0; i2 < NN3[1] ; i2++) {
      size_t k = nt3.at({i0, i1,i2}) ;
      nt3.row[k] = datain[k] ;
    }
    }
    }
    nt1.fill();
    nt2.fill();
    nt3.fill();

  }

  // Close the ntuple
  nt1.close();
  nt2.close();
  nt3.close();

  // close the filespace
  H5Fclose(file_id);
  return 0;
}
