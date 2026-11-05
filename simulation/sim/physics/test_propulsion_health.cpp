#include "propulsion_health.hpp"
#include <cassert>
#include <iostream>

int main() {
  using namespace x500;
  Powertrain core;
  core.step(.02, {.5,.5,.5,.5}, true);
  const auto state = core.state;
  const auto initial = PropulsionSnapshot::from(core.p);
  auto target = initial;
  target.resistance *= 1.08;
  target.friction += 8e-6;
  target.kt *= .97;
  target.kq *= 1.03;
  PropulsionTransition transition;
  transition.configure(core, target, 20);
  for (double seconds : {0., 5., 10., 20., 25.}) {
    transition.apply(core, seconds);
    double f = std::clamp(seconds/20., 0., 1.);
    assert(transition.fraction == f);
    assert(std::abs(core.p.resistance-(initial.resistance+f*(target.resistance-initial.resistance)))<1e-14);
    assert(std::abs(core.p.friction-(initial.friction+f*(target.friction-initial.friction)))<1e-18);
    assert(std::abs(core.p.kt_prop-(initial.kt+f*(target.kt-initial.kt)))<1e-18);
    assert(std::abs(core.p.kq_prop-(initial.kq+f*(target.kq-initial.kq)))<1e-20);
    assert(core.state.omega == state.omega && core.state.motor_temperature == state.motor_temperature);
    assert(core.state.soc == state.soc && core.state.polarization == state.polarization);
    assert(core.state.bus_energy == state.bus_energy && core.state.loss_energy == state.loss_energy);
  }
  initial.install(core.p);
  transition.configure(core, target, 1);
  for (int i=0;i<2000;++i) {
    transition.apply(core, i*.001);
    core.step(.001, {.5,.5,.5,.5}, true);
  }
  assert(std::abs(core.energy_residual()) < .001*core.state.bus_energy);
  assert(std::abs(core.battery_energy_residual()) < 1e-5*core.state.battery_source_energy);
  bool rejected=false;
  try { transition.configure(core, initial, 20); }
  catch(const std::invalid_argument &) { rejected=true; }
  assert(rejected);
  std::cout << "PASS joint propulsion interpolation, continuous state and energy accounting\n";
}
