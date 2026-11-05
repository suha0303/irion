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
  double max_substep = 0.00025;
  std::array<double, 31> ocv_cell = profile::ocv_cell;
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
  // Signed changes in accessible charge/energy caused by prescribed health changes.
  double battery_parameter_charge_c = 0, battery_parameter_energy_j = 0,
         battery_rc_parameter_work_j = 0;
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
    double index = std::clamp(soc, 0., 1.) * 30;
    int lo = std::min(29, static_cast<int>(index));
    return p.series_cells * (p.ocv_cell[lo] +
                (index - lo) * (p.ocv_cell[lo + 1] - p.ocv_cell[lo]));
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
