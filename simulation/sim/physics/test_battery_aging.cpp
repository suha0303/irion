#include "battery_aging.hpp"
#include "control.hpp"
#include <cassert>
#include <iostream>

using namespace x500;
static bool near(double a, double b, double tolerance = 1e-8) {
  return std::abs(a - b) < tolerance;
}
int main(int argc, char **argv) {
  Parameters p;
  for (std::size_t k = 0; k < 31; ++k)
    p.ocv_cell[k] = 3.0 + 1.2 * k / 30;
  if (argc == 3) {
    BatterySnapshot::read(argv[1]).install(p);
  }
  Powertrain core(p);
  core.state.soc = .8;
  auto initial = BatterySnapshot::from(core.p);
  auto target = initial;
  target.r0 *= 1.3;
  target.r1 *= 1.4;
  target.c1 *= 1.2;
  target.capacity *= .9;
  for (double &v : target.ocv)
    v -= .02;
  if (argc == 3)
    target = BatterySnapshot::read(argv[2]);

  // The measured profile still feeds the existing motor/battery bus solve.
  RotorValues low{}, high{};
  low.fill(.2);
  high.fill(.5);
  auto a = core.solve(core.state, low, true);
  auto b = core.solve(core.state, high, true);
  assert(b.current > a.current);
  assert(b.voltage < a.voltage);
  // Compare the same propeller speed/load, as a flight controller would demand.
  // Raising winding resistance or bearing drag needs more duty and battery
  // current. The bus voltage, charge use and heat must respond through the same
  // measured battery, without changing its health parameters.
  auto operating_point = [](const Parameters &parameters) {
    Powertrain motor(parameters);
    motor.state.soc = .8;
    motor.state.omega.fill(500.0);
    double lo = 0, hi = 1;
    for (int k = 0; k < 60; ++k) {
      RotorValues duty{};
      duty.fill((lo + hi) / 2);
      if (motor.solve(motor.state, duty, true).acceleration[0] > 0)
        hi = duty[0];
      else
        lo = duty[0];
    }
    RotorValues duty{};
    duty.fill((lo + hi) / 2);
    auto output = motor.solve(motor.state, duty, true);
    assert(std::abs(output.acceleration[0]) < 1e-5);
    motor.step(.01, duty, true, 298.15);
    return std::make_pair(motor, duty[0]);
  };
  for (auto battery : {initial, target}) {
    Parameters measured = p;
    battery.install(measured);
    auto healthy = operating_point(measured);
    for (bool winding : {false, true}) {
      Parameters damaged = measured;
      if (winding) damaged.resistance *= 1.2;
      else damaged.friction += .00004;
      auto degraded = operating_point(damaged);
      assert(degraded.second > healthy.second);
      assert(degraded.first.state.charge_c > healthy.first.state.charge_c);
      assert(degraded.first.state.battery_heat_energy > healthy.first.state.battery_heat_energy);
      assert(degraded.first.state.soc < healthy.first.state.soc);
      assert(BatterySnapshot::from(degraded.first.p).ocv == battery.ocv);
      assert(degraded.first.p.capacity_ah == battery.capacity);
    }
  }
  for (int k = 0; k < 200; ++k)
    core.step(.001, high, true, 298.15);
  assert(core.state.soc < .8);

  BatteryTransition transition;
  transition.configure(core, target, 2.0);
  double energy_origin = accessible_battery_energy(core) + core.state.battery_terminal_energy +
                         core.state.battery_heat_energy - core.state.battery_parameter_energy_j;
  double charge_origin = 3600 * core.p.capacity_ah * core.state.soc + core.state.charge_c -
                         core.state.battery_parameter_charge_c;
  double soc = core.state.soc, vp = core.state.polarization;
  double before_residual = core.battery_energy_residual();
  transition.apply(core, .5);
  assert(near(transition.fraction, .25));
  assert(near(core.p.r0, initial.r0 + .25 * (target.r0 - initial.r0)));
  assert(near(core.p.capacity_ah, initial.capacity + .25 * (target.capacity - initial.capacity)));
  assert(core.state.soc == soc && core.state.polarization == vp);
  assert(near(core.battery_energy_residual(), before_residual));
  for (int k = 501; k <= 2000; ++k) {
    transition.apply(core, k * .001);
    core.step(.001, high, true, 298.15);
  }
  assert(transition.fraction == 1);
  assert(near(core.p.r0, target.r0));
  assert(near(core.p.r1, target.r1));
  assert(near(core.p.c1, target.c1));
  assert(near(core.p.capacity_ah, target.capacity));
  for (std::size_t k = 0; k < 31; ++k)
    assert(near(core.p.ocv_cell[k], target.ocv[k]));
  double energy = accessible_battery_energy(core) + core.state.battery_terminal_energy +
                  core.state.battery_heat_energy - core.state.battery_parameter_energy_j;
  double charge = 3600 * core.p.capacity_ah * core.state.soc + core.state.charge_c -
                  core.state.battery_parameter_charge_c;
  assert(near(charge, charge_origin, 1e-6));
  assert(near(energy, energy_origin, .05));
  assert(std::abs(core.battery_energy_residual()) < .05);
  auto end = BatterySnapshot::from(core.p);
  for (int k = 0; k < 1000; ++k)
    core.step(.001, high, true, 298.15);
  assert(core.p.r0 == end.r0 && core.p.r1 == end.r1 && core.p.c1 == end.c1 &&
         core.p.capacity_ah == end.capacity && core.p.ocv_cell == end.ocv);

  Changes changes;
  for (int field : {0, 2}) {
    bool rejected = false;
    try { changes.enqueue({1, field, .05, 0}); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
  }
  assert(changes.enqueue({2, 1, .00001, 0}));
  changes.apply(core, 0);
  assert(core.p.r0 == end.r0 && core.p.r1 == end.r1);
  std::cout << "battery snapshots, linear transition, motor feedback, charge/energy ledgers passed\n";
}
