#pragma once
#include <cstdint>
#include <stdexcept>
namespace x500 {
struct JointWeatherClock {
  int64_t origin_us = -1, notice_us = -1;
  void notice(int64_t origin, int64_t current) {
    if (origin < 60000000 || current < origin || current - origin > 500000 ||
        (origin_us >= 0 && origin != origin_us))
      throw std::invalid_argument("joint weather origin/notice latency");
    origin_us = origin; notice_us = current;
  }
  double phase_seconds(int64_t current) const {
    if (origin_us < 0 || current <= origin_us + 1000000) return 0.;
    return static_cast<double>(current - origin_us - 1000000) / 1e6;
  }
};
} // namespace x500
