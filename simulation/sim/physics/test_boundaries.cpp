#include "aerodynamics.hpp"
#include "command.hpp"
#include "control.hpp"
#include "sensor.hpp"
#include <iostream>
using namespace x500;
void check(bool value, const char *name) {
  if (!value)
    throw std::runtime_error(name);
  std::cout << "PASS " << name << '\n';
}
int main() {
  CommandGate g;
  Four duty{500, 500, 500, 500};
  auto accept = [&](uint64_t seq, int64_t sample, int64_t now) {
    return g.accept("x500.esc.request/1", "FR,RL,FL,RR",
                    "post_mapping_count_0_1000", seq, sample, now, duty);
  };
  check(accept(0, 1000000, 1000000) == "accepted",
        "first command and output map");
  check(g.duty[0] == .5, "count maps exactly once");
  check(accept(0, 1000000, 1001000) == "sequence_or_time_regression",
        "duplicate sequence rejected");
  check(accept(1, 0, 1000000) == "stale_or_future_sample",
        "old sample rejected");
  check(accept(1, 1100000, 1000000) == "stale_or_future_sample",
        "future sample rejected");
  check(g.accept("x500.esc.request/1", "FR,FL,RR,RL",
                 "post_mapping_count_0_1000", 1, 1001000, 1001000,
                 duty) == "protocol_mismatch",
        "motor order validated");
  check(g.accept("x500.esc.request/1", "FR,RL,FL,RR", "rpm", 1, 1001000,
                 1001000, duty) == "protocol_mismatch",
        "semantic validated");
  check(g.fresh(1500000) && !g.fresh(1500001), "timeout exact boundary");
  auto copy = g;
  check(g.fresh(1000000) == copy.fresh(1000000),
        "paused simulation clock freezes freshness");
  Changes changes;
  bool rejected = false;
  try {
    changes.enqueue({1, 2, 0, 0});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "dynamic R1 zero rejected before state mutation");
  check(changes.enqueue({2, 2, .02, 1000}), "positive R1 accepted");
  Powertrain p;
  p.state.polarization = .2;
  auto before = p.state;
  changes.apply(p, 1000);
  check(p.state.polarization == before.polarization &&
            p.state.soc == before.soc &&
            p.state.battery_heat_energy == before.battery_heat_energy,
        "R1 positive change preserves RC state and energy");
  BatterySensor a(71), b(71);
  a.cfg.voltage_noise_std_v = b.cfg.voltage_noise_std_v = 0;
  a.cfg.current_noise_std_a = b.cfg.current_noise_std_a = 0;
  a.cfg.voltage_bias_v = .15;
  a.cfg.current_bias_a = .3;
  a.cfg.delay_s = .05;
  a.cfg.quantization_v = .1;
  check(a.step(0, 12, 3).empty(), "sensor delay holds sample");
  auto samples = a.step(.05, 19, 50);
  check(samples.size() == 1 && std::abs(samples[0].voltage - 12.2) < 1e-10 &&
            std::abs(samples[0].current - 3.3) < 1e-10 &&
            samples[0].sample_s == 0,
        "bias quantization and original sample time");
  a.cfg.voltage_bias_v = 100;
  samples = a.step(.1, 12, 3);
  samples = a.step(.15, 12, 3);
  check(samples[0].voltage == 20 && samples[0].clipped,
        "sensor saturation marked");
  b.cfg.drop_begin_s = .2;
  b.cfg.drop_end_s = .5;
  unsigned n = 0;
  for (int i = 0; i < 10; ++i)
    n += b.step(i * .1, 12, 3).size();
  check(n == 7 && b.dropped == 3, "missing sensor samples preserve gap");
  BatterySensor noise1(83), noise2(83);
  double sum = 0, sq = 0;
  unsigned count = 0;
  for (int k = 0; k < 50000; ++k) {
    auto x = noise1.step(k * .1, 12, 3), y = noise2.step(k * .1, 12, 3);
    if (x.size() != y.size())
      throw std::runtime_error("seed sample cadence");
    if (!x.empty()) {
      if (x[0].voltage != y[0].voltage)
        throw std::runtime_error("seed reproduction");
      double e = x[0].voltage - 12;
      sum += e;
      sq += e * e;
      ++count;
    }
  }
  check(std::abs(sum / count) < .0002 && std::abs(sq / count - .0001) < .000005,
        "white noise mean variance seed");
  BatterySensor drift1(91), drift2(91);
  drift1.cfg.voltage_noise_std_v = drift2.cfg.voltage_noise_std_v = 0;
  drift1.cfg.drift_std_v = drift2.cfg.drift_std_v = .04;
  drift1.cfg.drift_tau_s = drift2.cfg.drift_tau_s = 1;
  sum = sq = 0;
  double lag = 0, previous = 0;
  count = 0;
  for (int k = 0; k < 100000; ++k) {
    auto x = drift1.step(k * .1, 12, 3), y = drift2.step(k * .1, 12, 3);
    if (x.size() != y.size())
      throw std::runtime_error("drift cadence");
    if (x.empty())
      continue;
    if (x[0].voltage != y[0].voltage)
      throw std::runtime_error("drift seed reproduction");
    double e = x[0].voltage - 12;
    if (k > 1000) {
      sum += e;
      sq += e * e;
      lag += e * previous;
      ++count;
    }
    previous = e;
  }
  check(std::abs(sum / count) < .003 && std::abs(sq / count - .0016) < .00015 &&
            std::abs(lag / sq - std::exp(-.1)) < .02,
        "OU drift mean variance autocorrelation and seed");
  return 0;
}
