#pragma once
#include "powertrain.hpp"
#include <fstream>

namespace x500 {
struct BatterySnapshot {
  double r0{}, r1{}, c1{}, capacity{};
  std::array<double, 31> ocv{};

  static BatterySnapshot from(const Parameters &p) {
    return {p.r0, p.r1, p.c1, p.capacity_ah, p.ocv_cell};
  }
  void validate() const {
    for (double v : {r0, r1, c1, capacity})
      if (!std::isfinite(v) || v <= 0)
        throw std::invalid_argument("invalid measured battery snapshot");
    double previous = 0;
    for (double v : ocv) {
      if (!std::isfinite(v) || v < previous || v <= 0 || v >= 5)
        throw std::invalid_argument("invalid measured battery OCV");
      previous = v;
    }
  }
  static BatterySnapshot read(const char *path) {
    std::ifstream input(path);
    BatterySnapshot s;
    input >> s.r0 >> s.r1 >> s.c1 >> s.capacity;
    for (double &v : s.ocv)
      input >> v;
    if (!input)
      throw std::invalid_argument("incomplete battery snapshot file");
    std::string extra;
    if (input >> extra)
      throw std::invalid_argument("trailing battery snapshot data");
    s.validate();
    return s;
  }
  void install(Parameters &p) const {
    p.r0 = r0;
    p.r1 = r1;
    p.c1 = c1;
    p.capacity_ah = capacity;
    p.ocv_cell = ocv;
  }
};

// Stored accessible chemical energy for the same piecewise-linear OCV as core.ocv().
inline double accessible_battery_energy(const Powertrain &core) {
  const double z = std::clamp(core.state.soc, 0., 1.);
  double integral = 0;
  for (std::size_t k = 0; k < 30; ++k) {
    double start = static_cast<double>(k) / 30;
    double width = std::clamp(z - start, 0., 1. / 30);
    double a = core.p.ocv_cell[k];
    double slope = 30 * (core.p.ocv_cell[k + 1] - a);
    integral += a * width + .5 * slope * width * width;
  }
  return 3600 * core.p.capacity_ah * core.p.series_cells * integral +
         .5 * core.p.c1 * core.state.polarization * core.state.polarization;
}

class BatteryTransition {
  BatterySnapshot initial{}, target{};
  double duration = 0;

public:
  double fraction = 0;
  void configure(const Powertrain &core, const BatterySnapshot &end, double seconds) {
    if (!std::isfinite(seconds) || seconds <= 0 || seconds > 1800)
      throw std::invalid_argument("invalid battery transition duration or initial model");
    initial = BatterySnapshot::from(core.p);
    initial.validate();
    end.validate();
    target = end;
    duration = seconds;
    fraction = 0;
  }
  void apply(Powertrain &core, double elapsed) {
    if (duration <= 0 || !std::isfinite(elapsed))
      throw std::invalid_argument("unconfigured battery transition");
    const double next = std::clamp(elapsed / duration, 0., 1.);
    if (next < fraction)
      throw std::invalid_argument("battery transition cannot move backwards");
    if (next == fraction)
      return;
    auto mix = [next](double a, double b) { return a + next * (b - a); };
    BatterySnapshot p{mix(initial.r0, target.r0), mix(initial.r1, target.r1),
                      mix(initial.c1, target.c1), mix(initial.capacity, target.capacity), {}};
    for (std::size_t k = 0; k < 31; ++k)
      p.ocv[k] = mix(initial.ocv[k], target.ocv[k]);
    // This is an imposed health transition, not an electrochemical aging ODE.
    // Preserve fractional SOC and Vp; account explicitly for inaccessible charge
    // and changed chemical/RC energy instead of silently creating/destroying them.
    double before = accessible_battery_energy(core);
    core.state.battery_parameter_charge_c +=
        3600 * core.state.soc * (p.capacity - core.p.capacity_ah);
    core.state.battery_rc_parameter_work_j +=
        .5 * (p.c1 - core.p.c1) * core.state.polarization * core.state.polarization;
    p.install(core.p);
    core.state.battery_parameter_energy_j += accessible_battery_energy(core) - before;
    fraction = next;
  }
};
} // namespace x500
