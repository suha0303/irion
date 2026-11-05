#pragma once
#include "powertrain.hpp"
#include <fstream>

namespace x500 {
struct PropulsionSnapshot {
  double kt{}, kq{}, friction{}, coulomb{}, efficiency{}, resistance{};

  static PropulsionSnapshot from(const Parameters &p) {
    return {p.kt_prop, p.kq_prop, p.friction, p.coulomb,
            p.esc_efficiency, p.resistance};
  }
  void validate() const {
    for (double v : {kt, kq, efficiency, resistance})
      if (!std::isfinite(v) || v <= 0)
        throw std::invalid_argument("invalid propulsion snapshot");
    if (!std::isfinite(friction) || friction < 0 ||
        !std::isfinite(coulomb) || coulomb < 0 || efficiency > 1)
      throw std::invalid_argument("invalid propulsion loss parameters");
  }
  static PropulsionSnapshot read(const char *path) {
    PropulsionSnapshot s;
    std::ifstream file(path);
    if (!(file >> s.kt >> s.kq >> s.friction >> s.coulomb >>
          s.efficiency >> s.resistance))
      throw std::invalid_argument("invalid propulsion snapshot file");
    std::string extra;
    if (file >> extra)
      throw std::invalid_argument("unexpected propulsion snapshot values");
    s.validate();
    return s;
  }
  void install(Parameters &p) const {
    validate();
    p.kt_prop = kt;
    p.kq_prop = kq;
    p.friction = friction;
    p.coulomb = coulomb;
    p.esc_efficiency = efficiency;
    p.resistance = resistance;
  }
};

class PropulsionTransition {
  PropulsionSnapshot initial{}, target{};
  double seconds = 0;
public:
  double fraction = 0;
  void configure(const Powertrain &core, const PropulsionSnapshot &end, double duration) {
    if (!std::isfinite(duration) || duration <= 0 || duration > 1800)
      throw std::invalid_argument("positive propulsion transition duration required");
    initial = PropulsionSnapshot::from(core.p);
    initial.validate();
    end.validate();
    if (end.resistance < initial.resistance || end.friction < initial.friction ||
        end.kt > initial.kt || end.kq < initial.kq ||
        end.coulomb != initial.coulomb || end.efficiency != initial.efficiency)
      throw std::invalid_argument("inconsistent propulsion health endpoints");
    target = end;
    seconds = duration;
    fraction = 0;
  }
  void apply(Powertrain &core, double elapsed) {
    if (seconds <= 0 || !std::isfinite(elapsed))
      throw std::invalid_argument("unconfigured propulsion transition");
    const double next_fraction = std::clamp(elapsed / seconds, 0., 1.);
    if (next_fraction < fraction)
      throw std::invalid_argument("propulsion transition cannot move backwards");
    fraction = next_fraction;
    const auto mix = [this](double a, double b) { return a + fraction * (b - a); };
    PropulsionSnapshot next{mix(initial.kt, target.kt), mix(initial.kq, target.kq),
        mix(initial.friction, target.friction), initial.coulomb, initial.efficiency,
        mix(initial.resistance, target.resistance)};
    // Rotor inertia stays fixed, so speed, temperature and kinetic energy are continuous.
    next.install(core.p);
  }
};
} // namespace x500
