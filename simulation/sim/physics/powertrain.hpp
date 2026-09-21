#pragma once
#ifdef DRONE_PROFILE_HEADER
#include DRONE_PROFILE_HEADER
#else
#include "vehicle_generated.hpp"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace x500 {
inline constexpr std::size_t rotor_count = profile::spins.size();
using RotorValues = std::array<double, rotor_count>;
using Four = RotorValues;
struct Parameters {
#ifdef CAPSTONE_PROFILE_GENERIC
  double ke = profile::motor_ke, resistance = profile::motor_resistance;
  double inertia = profile::rotor_inertia, friction = profile::friction,
         coulomb = profile::coulomb;
  double kt_prop = profile::kt_prop, kq_prop = profile::kq_prop;
  double motor_limit = profile::motor_limit, dc_limit = profile::dc_limit,
         esc_efficiency = profile::esc_efficiency;
  double r0 = profile::r0, r1 = profile::r1, c1 = profile::c1,
         capacity_ah = profile::capacity_ah;
  double auxiliary_a = profile::auxiliary_a,
         motor_heat_capacity = profile::motor_heat_capacity,
         motor_cooling = profile::motor_cooling;
  double battery_heat_capacity = profile::battery_heat_capacity,
         battery_cooling = profile::battery_cooling;
  unsigned series_cells = profile::series_cells;
#else
  double ke = 60.0 / (2.0 * M_PI * 920.0), resistance = 0.12;
  double inertia = profile::rotor_inertia, friction = 2e-6, coulomb = 0.002;
  double kt_prop = 8.54858e-6, kq_prop = 1.36777e-7;
  double motor_limit = 40, dc_limit = 20, esc_efficiency = 0.95;
  double r0 = 0.04, r1 = 0.015, c1 = 2000, capacity_ah = 6;
  double auxiliary_a = 0.3, motor_heat_capacity = 35, motor_cooling = 0.8;
  double battery_heat_capacity = 650, battery_cooling = 1.5;
  unsigned series_cells = 4;
#endif
  double max_substep = 0.00025;
#ifdef CAPSTONE_PROFILE_GENERIC
  std::array<double, 31> ocv_cell = profile::ocv_cell;
  bool calibrated = true;
#else
  std::array<double, 31> ocv_cell{};
  bool calibrated = false;
#endif
  void validate() const;
};
struct State {
  RotorValues omega{}, motor_temperature = [] {
    RotorValues temperatures{};
    temperatures.fill(298.15);
    return temperatures;
  }();
  double soc = 0.9, polarization = 0, temperature = 298.15;
  double bus_energy = 0, shaft_energy = 0, loss_energy = 0, charge_c = 0;
  double battery_source_energy = 0, battery_terminal_energy = 0,
         battery_heat_energy = 0;
};
struct Output {
  RotorValues motor_current{}, dc_current{}, thrust{}, torque{}, acceleration{};
  double voltage = 0, current = 0, residual = 0;
  double shaft_power = 0, loss_power = 0;
  int iterations = 0;
  enum Status { normal, brownout, depleted } status = normal;
};
struct Derivative {
  RotorValues omega{}, temperature{};
  double soc{}, vp{}, temperature_pack{};
};
class Powertrain {
public:
  Parameters p;
  State state;
  explicit Powertrain(const Parameters &parameters = {}) : p(parameters) {
    p.validate();
  }
  double ocv(double soc) const {
    if (p.calibrated) {
      double index = std::clamp(soc, 0., 1.) * 30;
      int lo = std::min(29, static_cast<int>(index));
      return p.series_cells * (p.ocv_cell[lo] +
                  (index - lo) * (p.ocv_cell[lo + 1] - p.ocv_cell[lo]));
    }
    // Prior only: public-data calibration must replace this before acceptance.
    constexpr std::array<double, 6> z{0, .1, .3, .6, .9, 1};
    constexpr std::array<double, 6> v{3.2, 3.5, 3.65, 3.8, 4.05, 4.2};
    soc = std::clamp(soc, 0., 1.);
    for (unsigned i = 1; i < z.size(); ++i)
      if (soc <= z[i])
        return p.series_cells * (v[i - 1] +
                    (v[i] - v[i - 1]) * (soc - z[i - 1]) / (z[i] - z[i - 1]));
    return 4.2 * p.series_cells;
  }
  Output evaluate(double voltage, const State &s, const RotorValues &duty,
                  bool armed) const;
  Output solve(const State &s, const RotorValues &duty, bool armed) const;
  Output step(double dt, const RotorValues &duty, bool armed,
              double air_temperature = 298.15);
  double energy_residual() const;
  double battery_energy_residual() const;

private:
  Derivative derivative(const State &s, const Output &o, double ambient) const;
  static State advance(const State &s, const Derivative &d, double dt);
};
inline double rc_step(double vp, double current, double r, double c,
                      double dt) {
  if (r == 0)
    return 0;
  double a = std::exp(-dt / (r * c));
  return a * vp + r * (1 - a) * current;
}
} // namespace x500
