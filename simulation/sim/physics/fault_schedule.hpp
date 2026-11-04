#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace x500 {
// The runner supplies the observed airborne/path origin once. The configured
// physical fault remains disabled until that origin is acknowledged.
struct MissionFaultSchedule {
  bool enabled = false;
  long long delay_us = -1, origin_us = -1, onset_us = -1;

  void configure(double seconds, double boot_onset, const std::string &kind) {
    if (enabled || !std::isfinite(seconds) || seconds <= 0 || seconds > 1800 ||
        !std::isfinite(boot_onset) || boot_onset >= 0 ||
        (kind != "r0" && kind != "friction"))
      throw std::invalid_argument("invalid deferred physical fault configuration");
    const auto delay = std::llround(seconds * 1e6);
    if (delay <= 0 || std::abs(seconds * 1e6 - delay) > 1e-6)
      throw std::invalid_argument("fault offset must resolve to integer microseconds");
    delay_us = delay;
    enabled = true;
  }

  long long schedule(double origin, long long current_us) {
    if (!enabled || !std::isfinite(origin) || origin < 0 || origin > 1e15 ||
        std::floor(origin) != origin || current_us < 0)
      throw std::invalid_argument("invalid mission fault origin");
    const auto requested = static_cast<long long>(origin);
    if (origin_us >= 0) {
      if (requested != origin_us)
        throw std::invalid_argument("mission fault origin already fixed");
      return onset_us;
    }
    const auto onset = requested + delay_us;
    if (requested > current_us || onset <= current_us || onset > 1000000000000000LL)
      throw std::invalid_argument("mission fault must be scheduled in the future");
    origin_us = requested;
    onset_us = onset;
    return onset;
  }
};
} // namespace x500
