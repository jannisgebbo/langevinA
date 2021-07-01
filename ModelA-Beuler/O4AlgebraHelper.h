#ifndef O4ALGEBRAHELPER
#define O4ALGEBRAHELPER


#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscts.h>

#include <array>

//For the generator and other O4 conventions, see Derek's notes.

class O4AlgebraHelper{
public:
  static void O4Rotation(PetscScalar* V, PetscScalar* A, PetscScalar* phi); //In place O4 rotation specified by a vector and axial charge
};

class Vector {
public:

  Vector(PetscScalar v1, PetscScalar v2, PetscScalar v3):v{v1,v2,v3}{}
  Vector():v{0,0,0}{}

  PetscScalar norm() const
  {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  PetscScalar& operator[](int i){return v[i];}

  const PetscScalar& operator[](int i) const{
    return v[i];
  }


private:
  std::array<PetscScalar, 3> v;
};



Vector operator+(const Vector& a, const Vector& b){
    Vector c;
    for(int i = 0; i<3; ++i){
      c[i] = a[i] + b[i];
    }
    return c;
}

Vector operator*(PetscScalar a, const Vector& b){
    Vector c;
    for(int i = 0; i<3; ++i){
      c[i] = a * b[i];
    }
    return c;


}

Vector cross(const Vector& a, const Vector& b){
  return Vector(a[1] * b[2] - b[1] * a[2], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - b[0] * a[1]);
}

PetscScalar dot(const Vector& a, const Vector& b){
  PetscScalar res = 0;
  for(int i = 0; i<3; ++i)
  {
    res += a[i] * b[i];
  }
  return res;
}

class Quaternion{
public:
  Quaternion(PetscScalar w0, const Vector& wi):v0(w0), vi(wi){}
  Quaternion(){}

  PetscScalar& operator[](int i){
    if(i>0) return vi[i-1];
    else return v0;
  }
  const PetscScalar& operator[](int i) const
  {
    if(i>0) return vi[i-1];
    else return v0;
  }

  const Vector& getV() const{
    return vi;
  }

  Vector& getV() {
    return vi;
  }

private:
  PetscScalar v0;
  Vector vi;
};

Quaternion operator*(const Quaternion& v1, const Quaternion& v2 )
{
  Quaternion res;

  res[0] = v1[0] * v2[0] - dot(v1.getV(), v2.getV());
  res.getV() = v1[0] * v2.getV() + v2[0] * v1.getV() + cross(v1.getV(), v2.getV());
  return res;
}


Quaternion exp(const Vector& v){
  PetscScalar norm = v.norm();
  return Quaternion(cos(norm), - (sin(norm) / norm) * v);
}


#endif
