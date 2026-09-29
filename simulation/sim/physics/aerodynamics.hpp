#pragma once
#include "powertrain.hpp"
namespace x500 {
using Vec3 = std::array<double, 3>;
constexpr auto rotor_positions_flu = profile::positions;
constexpr auto spin = profile::spins;
inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
inline Vec3 enu_to_ned(const Vec3 &v) { return {v[1], v[0], -v[2]}; }
inline Vec3 flu_to_frd(const Vec3 &v) { return {v[0], -v[1], -v[2]}; }
struct Wrench {
  Vec3 force{}, moment{};
};
inline Wrench wrench(const Parameters &p, const State &s, const Output &o,
                     const Vec3 &air, const Vec3 &rates) {
  Wrench out;
  for (std::size_t i = 0; i < rotor_count; ++i) {
    auto v = cross(rates, rotor_positions_flu[i]);
    for (int k = 0; k < 3; ++k)
      v[k] += air[k];
#ifdef CAPSTONE_PROFILE_GENERIC
    constexpr double rotor_drag = profile::rotor_drag_coefficient;
#else
    constexpr double rotor_drag = 8.06428e-5;
#endif
    Vec3 f{-rotor_drag * s.omega[i] * v[0], -rotor_drag * s.omega[i] * v[1],
           o.thrust[i]};
    auto lever = cross(rotor_positions_flu[i], f);
    auto gyro = cross(rates, {0, 0, spin[i] * p.inertia * s.omega[i]});
    for (int k = 0; k < 3; ++k) {
      out.force[k] += f[k];
      out.moment[k] += lever[k] - gyro[k];
    }
    out.moment[2] -= spin[i] * (o.torque[i] + p.inertia * o.acceleration[i]);
  }
  for (int k = 0; k < 3; ++k)
#ifdef CAPSTONE_PROFILE_GENERIC
    out.force[k] -= .5 * 1.225 * profile::body_drag_area * std::abs(air[k]) * air[k];
#else
    out.force[k] -= .5 * 1.225 * .08 * std::abs(air[k]) * air[k];
#endif
  return out;
}
} // namespace x500
