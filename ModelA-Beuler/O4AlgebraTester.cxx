#include "O4AlgebraHelper.h"

#include <iostream>

int main(){

  Vector a(1,2,3);
  Vector b(4,6,7);

  Vector c = a + b;
  Vector d = 2 * b;

  for(int i=0; i<3; ++i){
    std::cout << c[i] << std::endl;
    std::cout << d[i] << std::endl;
  }

  auto q = exp(a);

  for(int i=0; i<4; ++i){
    std::cout << q[i] << std::endl;
  }

  Quaternion v1(0.9, Vector(0.6,0.3,0.5));
  Quaternion v2(0.1, Vector(0.2,0.6,0.9));

  auto v3 = v1 * v2;


  for(int i=0; i<4; ++i){
    std::cout << v3[i] << std::endl;
  }




  return 0;
}
