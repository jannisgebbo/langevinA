#include "ModelA.h"
#include <cmath>
#include <tuple>

// Rotate around Z-axis by phi (azimuthal), then around Y-axis by theta (polar)
std::tuple<double, double, double> rotateVector(double dx, double dy, double dz,
                                                double theta, double phi) {
  // First rotation: around Z-axis by phi (azimuthal angle)
  double cos_phi = std::cos(phi);
  double sin_phi = std::sin(phi);

  double x1 = dx * cos_phi - dy * sin_phi;
  double y1 = dx * sin_phi + dy * cos_phi;
  double z1 = dz;

  // Second rotation: around Y-axis by theta (polar angle)
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  double x2 = x1 * cos_theta + z1 * sin_theta;
  double y2 = y1;
  double z2 = -x1 * sin_theta + z1 * cos_theta;

  return std::make_tuple(x2, y2, z2);
}

void initialize_gaussians(G_node *node, const double &x, const double &y,
                          const double &z, ModelA *model, void *ctx) {
  auto &inputs = *reinterpret_cast<nlohmann::json *>(ctx);
  double sigmax = inputs["sigmax"];
  double sigmay = inputs["sigmay"];
  double sigmaz = inputs["sigmaz"];
  double A = inputs["amplitude"];
  data_node *u = reinterpret_cast<data_node *>(node);

  for (int L = 0; L < ModelAData::Ndof; L++) {
    double dx = x - model->data.LX / 2.0;
    double dy = y - model->data.LY / 2.0;
    double dz = z - model->data.LZ / 2.0;
    double theta = inputs["theta"];
    double phi = inputs["phi"];
    auto [xp, yp, zp] = rotateVector(dx, dy, dz, theta, phi);
    u->x[L] = A * exp(-xp * xp / (2. * sigmax * sigmax) -
                      yp * yp / (2. * sigmay * sigmay) -
                      zp * zp / (2. * sigmaz * sigmaz));
  }
}
// Routine that initializes the fields according to a wave with wave number k =
// 4pi/L and normalization phi^2 = init_amp = R
//
// if init_dim = 1 and standing_waves = false
// s_0 = R * cos(kx)
// s_1 = R * sin(kx)
// n_01 = k * chi = omega * chi
// rest 0
//
// if init_dim = 2
// s_0 = R * cos(kx) cos(ky)
// s_1 = R * sin(kx) cos(ky)
// s_2 = R * sin(ky)
// n_01 = k * chi = omega * chi
// n_02 = k * chi = omega * chi
// n_12 = k * chi = omega * chi
// rest 0
//
// if init_dim = 1 and standing_waves = true
// s_0 = R * cos(kx)
// s_1 = 0
// s_2 = R * sin(ky)
// n_01 = k * chi = omega * chi
// rest 0
void initialize_wave_spins(G_node *node, const double &x, const double &y,
                           const double &z, ModelA *model, void *ctx) {

  auto &inputs = *reinterpret_cast<nlohmann::json *>(ctx);
  PetscScalar *u = reinterpret_cast<PetscScalar *>(node);

  int init_dimension = inputs["init_dimension"].get<int>();
  bool standing_waves = inputs["standing_waves"].get<bool>();

  double init_amp = sqrt(model->data.f2());
  constexpr auto PI = 3.14159265358979323846;

  PetscScalar chi = model->data.acoefficients.chi;
  PetscReal wave_k = 4 * PI / model->data.LX;
  PetscReal argument = wave_k;

  for (int L = 0; L < ModelAData::Ndof; L++) {
    u[L] = 0;

    // field components s_a
    if (L < ModelAData::Nphi) {

      // s_0 component
      if (L == 0) {
        u[L] = init_amp;
        // 1d init cond.
        if (init_dimension == 1) {
          u[L] *= cos(argument * x);
        }
        // 2d init cond.
        else if (init_dimension == 2) {
          u[L] *= cos(argument * x) * cos(argument * y);
        }
      }
      // s_1 component
      else if (L == 1) {
        u[L] = init_amp;
        // 1d init cond.
        if (init_dimension == 1) {
          // if standing wave solution
          if (standing_waves) {
            u[L] *= 0.0;
          } else {
            u[L] *= sin(argument * x);
          }
        }
        // 2d init cond.
        else if (init_dimension == 2) {
          u[L] *= sin(argument * x) * cos(argument * y);
        }
      }
      // s_2 component
      else if (L == 2) {
        u[L] = init_amp;
        // 1d init cond.
        if (init_dimension == 1) {
          // if standing wave solution
          if (standing_waves) {
            u[L] *= sin(argument * x);
          } else {
            u[L] *= 0.0;
          }
        }
        // 2d init cond.
        else if (init_dimension == 2) {
          u[L] *= sin(argument * y);
        }
      }
    }

    // charges n_A (4,5,6) and n_V (7,8,9)
    else {
      // (n_A)_0 (or n_01)
      if (L == 4) {
        u[L] = wave_k * chi;
      }
      // (n_A)_0 (or n_01) OR (n_V)_2 (or n_12)
      else if (L == 5 || L == 9) {
        // 2d init cond.
        if (init_dimension == 2) {
          u[L] = wave_k * chi;
        }
      }
    }
  } // end of assignment for single point
}
