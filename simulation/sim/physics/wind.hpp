#pragma once
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
namespace x500 {
// Independent exact-discrete Ornstein-Uhlenbeck axes, not a Dryden transverse
// spectrum.
class Wind {
public:
  std::array<double, 3> gust{};
  std::mt19937 engine;
  std::normal_distribution<double> gaussian{0, 1};
  double sigma, tau;
  Wind(unsigned seed, double intensity, double correlation = 2)
      : engine(seed), sigma(intensity), tau(correlation) {
    if (sigma < 0 || !std::isfinite(sigma) || tau <= 0 || !std::isfinite(tau))
      throw std::invalid_argument("wind parameter");
    for (auto &v : gust)
      v = sigma * gaussian(engine); // stationary initial state
  }
  void step(double dt) {
    if (dt <= 0 || !std::isfinite(dt))
      throw std::invalid_argument("wind dt");
    double a = std::exp(-dt / tau), b = sigma * std::sqrt(1 - a * a);
    for (auto &v : gust)
      v = a * v + b * gaussian(engine);
  }
};
} // namespace x500
