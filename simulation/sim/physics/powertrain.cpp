#include "powertrain.hpp"
namespace x500 {
void Parameters::validate() const {
  for (double x :
       {ke, resistance, inertia, kt_prop, kq_prop, motor_limit, dc_limit, c1,
        capacity_ah, motor_heat_capacity, motor_cooling, battery_heat_capacity,
        battery_cooling, max_substep})
    if (!std::isfinite(x) || x <= 0)
      throw std::invalid_argument("positive finite parameter required");
  for (double x : {friction, coulomb, r0, r1, auxiliary_a})
    if (!std::isfinite(x) || x < 0)
      throw std::invalid_argument("nonnegative parameter required");
  if (!(esc_efficiency > 0 && esc_efficiency <= 1))
    throw std::invalid_argument("ESC efficiency");
  if (series_cells < 1 || series_cells > 16)
    throw std::invalid_argument("battery series cells outside PX4 range [1,16]");
}
Output Powertrain::evaluate(double voltage, const State &s, const RotorValues &duty,
                            bool armed) const {
  Output o;
  o.voltage = voltage;
  o.current = voltage > 0 ? p.auxiliary_a : 0;
  for (std::size_t i = 0; i < rotor_count; ++i) {
    double w = s.omega[i];
    double r = p.resistance * (1 + .00393 * (s.motor_temperature[i] - 298.15));
    double e = p.ke * w;
    double requested =
        armed && s.soc > 0 ? std::max(0., (duty[i] * voltage - e) / r) : 0;
    double budget = p.esc_efficiency * voltage * p.dc_limit;
    // Stable positive root of r*I^2 + e*I = budget.
    double power_limit =
        budget > 0 ? 2 * budget / (e + std::sqrt(e * e + 4 * r * budget)) : 0;
    double im = std::min({requested, p.motor_limit, power_limit});
    double pin = (e + r * im) * im;
    double idc = voltage > 0 ? pin / (p.esc_efficiency * voltage) : 0;
    double friction = p.friction * w + p.coulomb * std::tanh(w);
    double q = p.kq_prop * w * w;
    o.motor_current[i] = im;
    o.dc_current[i] = idc;
    o.current += idc;
    o.thrust[i] = p.kt_prop * w * w;
    o.torque[i] = q;
    o.acceleration[i] = (p.ke * im - q - friction) / p.inertia;
    o.shaft_power += q * w;
    o.loss_power += im * im * r + friction * w + voltage * idc - pin;
  }
  return o;
}
Output Powertrain::solve(const State &s, const RotorValues &duty, bool armed) const {
  double hi = std::max(0., ocv(s.soc) - s.polarization), lo = 0;
  if (s.soc <= 0 || hi <= p.r0 * p.auxiliary_a) {
    auto out = evaluate(0, s, duty, false);
    out.status = s.soc <= 0 ? Output::depleted : Output::brownout;
    return out;
  }
  Output out;
  for (int k = 1; k <= 64; ++k) {
    double v = (lo + hi) * .5;
    out = evaluate(v, s, duty, armed);
    out.residual = v + p.r0 * out.current + s.polarization - ocv(s.soc);
    out.iterations = k;
    if (std::abs(out.residual) < 1e-8)
      return out;
    if (out.residual > 0)
      hi = v;
    else
      lo = v;
  }
  throw std::runtime_error("bus solver failed to converge");
}
Derivative Powertrain::derivative(const State &s, const Output &o,
                                  double ambient) const {
  Derivative d;
  d.omega = o.acceleration;
  for (std::size_t i = 0; i < rotor_count; ++i) {
    double r = p.resistance * (1 + .00393 * (s.motor_temperature[i] - 298.15));
    double w = s.omega[i];
    double heat = o.motor_current[i] * o.motor_current[i] * r +
                  (p.friction * w + p.coulomb * std::tanh(w)) * w;
    d.temperature[i] =
        (heat - p.motor_cooling * (s.motor_temperature[i] - ambient)) /
        p.motor_heat_capacity;
  }
  d.soc = -o.current / (3600 * p.capacity_ah);
  d.vp = p.r1 > 0 ? (o.current - s.polarization / p.r1) / p.c1 : 0;
  double heat = o.current * o.current * p.r0 +
                (p.r1 > 0 ? s.polarization * s.polarization / p.r1 : 0);
  d.temperature_pack = (heat - p.battery_cooling * (s.temperature - ambient)) /
                       p.battery_heat_capacity;
  return d;
}
State Powertrain::advance(const State &s, const Derivative &d, double dt) {
  State n = s;
  for (std::size_t i = 0; i < rotor_count; ++i) {
    n.omega[i] = std::max(0., s.omega[i] + dt * d.omega[i]);
    n.motor_temperature[i] += dt * d.temperature[i];
  }
  n.soc = std::max(0., s.soc + dt * d.soc);
  n.polarization += dt * d.vp;
  n.temperature += dt * d.temperature_pack;
  return n;
}
Output Powertrain::step(double dt, const RotorValues &duty, bool armed,
                        double ambient) {
  if (!std::isfinite(dt) || dt <= 0 || dt > 0.1 || !std::isfinite(ambient) ||
      ambient < 200 || ambient > 400)
    throw std::invalid_argument("invalid timestep or temperature");
  for (double d : duty)
    if (!std::isfinite(d) || d < 0 || d > 1)
      throw std::invalid_argument("duty outside [0,1]");
  p.validate();
  int n = static_cast<int>(std::ceil(dt / p.max_substep));
  double h = dt / n;
  for (int j = 0; j < n; ++j) {
    if (p.r1 == 0)
      state.polarization = 0;
    auto first = solve(state, duty, armed);
    auto mid = advance(state, derivative(state, first, ambient), h * .5);
    auto out = solve(mid, duty, armed);
    auto next = advance(state, derivative(mid, out, ambient), h);
    next.bus_energy +=
        h * out.voltage * (out.current - (out.voltage > 0 ? p.auxiliary_a : 0));
    next.shaft_energy += h * out.shaft_power;
    next.loss_energy += h * out.loss_power;
    next.charge_c += h * out.current;
    next.battery_source_energy += h * ocv(mid.soc) * out.current;
    next.battery_terminal_energy += h * out.voltage * out.current;
    next.battery_heat_energy +=
        h * (out.current * out.current * p.r0 +
             (p.r1 > 0 ? mid.polarization * mid.polarization / p.r1 : 0));
    state = next;
  }
  return solve(state, duty, armed);
}
double Powertrain::energy_residual() const {
  double kinetic = 0;
  for (double w : state.omega)
    kinetic += .5 * p.inertia * w * w;
  return state.bus_energy - state.shaft_energy - state.loss_energy - kinetic;
}
double Powertrain::battery_energy_residual() const {
  double stored =
      p.r1 > 0 ? .5 * p.c1 * state.polarization * state.polarization : 0;
  return state.battery_source_energy - state.battery_terminal_energy -
         state.battery_heat_energy - stored;
}
} // namespace x500
