#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <stdexcept>
#include <vector>
namespace x500 {
struct BatterySensorConfig {
  double period_s = .1, delay_s = 0, voltage_bias_v = 0, current_bias_a = 0;
  double voltage_noise_std_v = .01, current_noise_std_a = .02;
  double drift_std_v = 0, drift_tau_s = 10, quantization_v = 0;
  double voltage_min = 0, voltage_max = 20, current_min = 0, current_max = 100;
  double drop_begin_s = -1, drop_end_s = -1;
};
struct BatterySample {
  double sample_s, release_s, voltage, current, physical_voltage,
      physical_current;
  bool clipped;
};
class BatterySensor {
  std::mt19937 rng;
  std::normal_distribution<double> normal{0, 1};
  std::deque<BatterySample> pending;
  double last = -1, drift = 0;

public:
  BatterySensorConfig cfg;
  unsigned dropped = 0;
  explicit BatterySensor(unsigned seed = 17) : rng(seed) {}
  std::vector<BatterySample> step(double now, double voltage, double current) {
    if (cfg.period_s <= 0 || cfg.delay_s < 0 || cfg.delay_s > .2 ||
        cfg.drift_tau_s <= 0)
      throw std::invalid_argument("battery sensor timing");
    if (last < 0 || now - last >= cfg.period_s - 1e-10) {
      double dt = last < 0 ? cfg.period_s : now - last;
      last = now;
      double a = std::exp(-dt / cfg.drift_tau_s);
      drift = a * drift + cfg.drift_std_v * std::sqrt(1 - a * a) * normal(rng);
      double v = voltage + cfg.voltage_bias_v + drift +
                 cfg.voltage_noise_std_v * normal(rng);
      double i =
          current + cfg.current_bias_a + cfg.current_noise_std_a * normal(rng);
      if (cfg.quantization_v > 0)
        v = std::round(v / cfg.quantization_v) * cfg.quantization_v;
      bool clipped = v < cfg.voltage_min || v > cfg.voltage_max ||
                     i < cfg.current_min || i > cfg.current_max;
      v = std::clamp(v, cfg.voltage_min, cfg.voltage_max);
      i = std::clamp(i, cfg.current_min, cfg.current_max);
      if (now >= cfg.drop_begin_s && now < cfg.drop_end_s)
        ++dropped;
      else
        pending.push_back(
            {now, now + cfg.delay_s, v, i, voltage, current, clipped});
    }
    std::vector<BatterySample> ready;
    while (!pending.empty() && pending.front().release_s <= now + 1e-10) {
      ready.push_back(pending.front());
      pending.pop_front();
    }
    return ready;
  }
};
} // namespace x500
