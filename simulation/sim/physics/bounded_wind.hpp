#pragma once
#include "wind.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace x500 {
// OU knots live in dimensionless speed/azimuth/elevation coordinates. The
// applied wind is a separate bounded C2 process, not Gaussian OU in m/s.
class BoundedWind {
public:
  std::string environment;
  double direction_deg, variation;
  Wind latent;
  std::array<double, 3> left{}, right{}, state{};
  uint64_t knot = 0;
  double last_time = -1;
  static constexpr double pi = 3.14159265358979323846;
  BoundedWind(unsigned seed, std::string name, double direction = 0,
              double intensity = 1)
      : environment(name), direction_deg(direction), variation(intensity),
        latent(seed, 1, 2) {
    if ((name != "calm" && name != "low" && name != "strong") ||
        !std::isfinite(direction) || direction < 0 || direction >= 360 ||
        !std::isfinite(intensity) || intensity < .5 || intensity > 1.5)
      throw std::invalid_argument("bounded wind profile");
    left = latent.gust;
    latent.step(1);
    right = latent.gust;
  }
  std::array<double, 3> transform(const std::array<double, 3> &z) const {
    for (double value : z)
      if (!std::isfinite(value))
        throw std::invalid_argument("finite bounded latent state required");
    if (environment == "calm")
      return {0, 0, 0};
    double center = environment == "low" ? 1.25 : 3.75;
    double speed = center + std::tanh(variation * z[0]);
    double azimuth = direction_deg * pi / 180 + pi / 3 * std::tanh(variation * z[1]);
    double elevation = pi / 12 * std::tanh(variation * z[2]);
    // Magnitude is speed. Preset bounds0.25..2.25 and2.75..4.75 leave
    // strict bin and numerical margins; no output-vector clipping is used.
    return {speed * std::cos(elevation) * std::sin(azimuth),
            speed * std::cos(elevation) * std::cos(azimuth),
            speed * std::sin(elevation)};
  }
  std::array<double, 3> sample(double time) {
    if (!std::isfinite(time) || time < 0 || time < last_time)
      throw std::invalid_argument("bounded wind monotonic clock");
    while (time >= static_cast<double>(knot + 1)) {
      left = right;
      latent.step(1);
      right = latent.gust;
      ++knot;
    }
    double u = time - static_cast<double>(knot);
    double blend = u * u * u * (10 + u * (-15 + 6 * u));
    for (int axis = 0; axis < 3; ++axis)
      state[axis] = left[axis] + blend * (right[axis] - left[axis]);
    last_time = time;
    return transform(state);
  }
};
} // namespace x500
