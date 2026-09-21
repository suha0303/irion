#include "aerodynamics.hpp"
#include "control.hpp"
#include "powertrain.hpp"
#include "wind.hpp"
#include <iostream>
#include <string>
using namespace x500;
void check(bool value, const std::string &name) {
  if (!value)
    throw std::runtime_error(name);
  std::cout << "PASS " << name << '\n';
}
Powertrain run(double step) {
  Parameters p;
  p.max_substep = step;
  Powertrain a(p);
  for (int i = 0; i < std::lround(2 / step); ++i)
    a.step(step, {.55, .55, .55, .55}, true);
  return a;
}
int main() {
  try {
    check(enu_to_ned({1, 2, 3}) == Vec3{2, 1, -3}, "ENU NED basis");
    check(flu_to_frd({1, 2, 3}) == Vec3{1, -2, -3}, "FLU FRD basis");
    check(enu_to_ned(enu_to_ned({1, 2, 3})) == Vec3{1, 2, 3},
          "coordinate roundtrip");
    Output symmetric;
    symmetric.thrust = {5, 5, 5, 5};
    symmetric.torque = {.1, .1, .1, .1};
    auto hover = wrench(Parameters{}, State{}, symmetric, {0, 0, 0}, {0, 0, 0});
    check(hover.force == Vec3{0, 0, 20} && hover.moment == Vec3{0, 0, 0},
          "symmetric hover wrench");
    symmetric.thrust = {5, 0, 0, 0};
    symmetric.torque = {.1, 0, 0, 0};
    auto single =
        wrench(Parameters{}, State{}, symmetric, {0, 0, 0}, {0, 0, 0});
    check(std::abs(single.moment[0] + .87) < 1e-12 &&
              std::abs(single.moment[1] + .87) < 1e-12 &&
              std::abs(single.moment[2] + .1) < 1e-12,
          "front right moment and yaw sign");
    symmetric.acceleration = {100, 0, 0, 0};
    auto accelerating =
        wrench(Parameters{}, State{}, symmetric, {0, 0, 0}, {0, 0, 0});
    check(std::abs(accelerating.moment[2] + .102) < 1e-12,
          "rotor inertia reaction included once");
    Changes changes;
    Powertrain controlled;
    check(changes.enqueue({1, 0, .08, 1000}), "request queued");
    check(!changes.enqueue({1, 0, .08, 1000}), "duplicate request idempotent");
    check(changes.apply(controlled, 999).empty(), "no early physical apply");
    auto receipts = changes.apply(controlled, 1000);
    check(receipts.size() == 1 && controlled.p.r0 == .08 &&
              receipts[0].applied_us == 1000,
          "step boundary applied timestamp");
    bool conflict = false;
    try {
      changes.enqueue({1, 0, .09, 1000});
    } catch (const std::invalid_argument &) {
      conflict = true;
    }
    check(conflict, "conflicting duplicate rejected");
    Wind wa(17, .4), wb(17, .4);
    for (int i = 0; i < 100; ++i) {
      wa.step(.01);
      wb.step(.01);
    }
    check(wa.gust == wb.gust, "wind seed exact reproduction");
    Wind saved = wa;
    wa.step(.01);
    saved.step(.01);
    check(wa.gust == saved.gust, "wind state restore including RNG");
    for (double dt : {.01, .02}) {
      Wind w(41, .4);
      double sum = 0, sq = 0, lag = 0, previous = w.gust[0];
      int n = static_cast<int>(20000 / dt);
      for (int i = 0; i < n; ++i) {
        w.step(dt);
        double x = w.gust[0];
        sum += x;
        sq += x * x;
        lag += x * previous;
        previous = x;
      }
      check(std::abs(sum / n) < .025 && std::abs(sq / n - .16) < .016,
            "wind mean/variance 20ks stationary test");
      check(std::abs(lag / sq - std::exp(-dt / 2)) < .005,
            "wind autocorrelation dt invariant");
    }
    check(std::abs(rc_step(0, 10, .02, 1000, 20) - .2 * (1 - std::exp(-1))) <
              1e-14,
          "analytic RC step");
    check(std::abs(rc_step(.2, 0, .02, 1000, 20) - .2 * std::exp(-1)) < 1e-14,
          "analytic RC relaxation");
    check(rc_step(.2, 1, 0, 1000, 1) == 0, "zero R1 branch");
    Powertrain a;
    auto s = a.state;
    auto o = a.solve(s, {.5, .5, .5, .5}, true);
    check(a.state.soc == s.soc && a.state.omega == s.omega,
          "voltage evaluation is pure");
    check(std::abs(o.residual) < 1e-6, "coupled bus residual");
    auto z = a.evaluate(0, s, {1, 1, 1, 1}, true);
    check(z.current == 0 && std::isfinite(z.loss_power), "zero bus finite");
    auto full = a.solve(s, {1, 1, 1, 1}, true);
    check(full.dc_current[0] <= a.p.dc_limit &&
              full.motor_current[0] <= a.p.motor_limit,
          "independent DC and motor limits");
    auto low = s;
    low.soc = .2;
    check(a.solve(low, {.5, .5, .5, .5}, true).voltage < o.voltage,
          "SOC affects bus at fixed command");
    auto b = run(.001), c = run(.0005), d = run(.00025);
    double e1 = std::abs(b.state.omega[0] - c.state.omega[0]);
    double e2 = std::abs(c.state.omega[0] - d.state.omega[0]);
    std::cout << "convergence omega " << e1 << ' ' << e2 << " energy "
              << b.energy_residual() << ' ' << c.energy_residual() << ' '
              << d.energy_residual() << '\n';
    auto ob = b.solve(b.state, {.55, .55, .55, .55}, true);
    auto oc = c.solve(c.state, {.55, .55, .55, .55}, true);
    auto od = d.solve(d.state, {.55, .55, .55, .55}, true);
    std::cout << "convergence voltage " << std::abs(ob.voltage - oc.voltage)
              << ' ' << std::abs(oc.voltage - od.voltage) << " current "
              << std::abs(ob.current - oc.current) << ' '
              << std::abs(oc.current - od.current) << '\n';
    check(std::abs(oc.voltage - od.voltage) <
              std::abs(ob.voltage - oc.voltage) * .5 + 2e-8,
          "RK2 voltage convergence with bus solve tolerance");
    check(std::abs(oc.current - od.current) <
              std::abs(ob.current - oc.current) * .5 + 1e-7,
          "RK2 current convergence with bus solve tolerance");
    check(e2 < e1 * .4, "RK2 dt convergence omega");
    check(std::abs(d.energy_residual()) < std::abs(c.energy_residual()) * .4,
          "energy residual convergence");
    check(std::abs(d.energy_residual()) < .001 * d.state.bus_energy,
          "motor energy balance <0.1 percent");
    check(std::abs((.9 - d.state.soc) * 3600 * d.p.capacity_ah -
                   d.state.charge_c) < 1e-7,
          "charge integral");
    check(std::abs(d.battery_energy_residual()) <
              1e-5 * d.state.battery_source_energy,
          "battery source-terminal-heat-RC energy balance");
    auto snapshot = d;
    d.step(.01, {.5, .5, .5, .5}, true);
    auto restored = snapshot;
    snapshot.step(.01, {.5, .5, .5, .5}, true);
    check(d.state.omega == snapshot.state.omega &&
              d.state.soc == snapshot.state.soc,
          "physical snapshot deterministic continuation");
    d = restored;
    auto steady = d.solve(d.state, {.55, .55, .55, .55}, true);
    check(std::abs(steady.acceleration[0]) < 1, "near steady torque balance");
    double kinetic_before = d.state.omega[0] * d.state.omega[0];
    d.step(.1, {0, 0, 0, 0}, false);
    check(d.state.omega[0] * d.state.omega[0] < kinetic_before,
          "coast kinetic decreases");
    d.state.soc = 0;
    check(d.solve(d.state, {1, 1, 1, 1}, true).status == Output::depleted,
          "depletion classification");
    a.state.polarization = 20;
    check(a.solve(a.state, {1, 1, 1, 1}, true).status == Output::brownout,
          "brownout classification");
    bool rejected = false;
    try {
      a.step(.01, {NAN, 0, 0, 0}, true);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    check(rejected, "reject invalid input");
    check(std::abs((4. / 2 * .02) * (2. / 4 * 1000) - .02 * 1000) < 1e-14,
          "cell pack tau invariant");
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
